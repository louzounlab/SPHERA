import ast
import pickle
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import find_rainbow_clique
from matplotlib.patches import Rectangle
import process_graph
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec


# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font_size = 25
font.set_size(font_size)

def combine(p, t_min, t_max, h_min, h_max):

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(6, 2)
    ax_h_0_no_gate = fig.add_subplot(gs[0, 0])
    ax_h_0_y_gate = fig.add_subplot(gs[0, 1])
    ax_no_gate_t = [fig.add_subplot(gs[i, 0]) for i in range(1, 4)]
    ax_no_gate_h = [fig.add_subplot(gs[i, 0]) for i in range(4, 6)]
    ax_y_gate_t = [fig.add_subplot(gs[i, 1]) for i in range(1, 4)]
    ax_y_gate_h = [fig.add_subplot(gs[i, 1]) for i in range(4, 6)]
    plot_function_prob(p, t_min, t_max, ax_h_0_no_gate, gate=False)
    plot_function_prob(p, t_min, t_max, ax_h_0_y_gate, gate=True)
    plot_function_prob_with_h(p, t_min, t_max, h_min, h_max, ax_no_gate_t + ax_no_gate_h,False)
    plot_function_prob_with_h(p, t_min, t_max, h_min, h_max, ax_y_gate_t + ax_y_gate_h, True)
    plt.tight_layout()
    plot_boxes(fig, [ax_h_0_no_gate, ax_h_0_y_gate], "red")
    plot_boxes(fig, ax_no_gate_h + ax_y_gate_h, "green")
    plot_boxes(fig, ax_no_gate_t + ax_y_gate_t, "blue")
    plt.savefig("Fig2.pdf", format="pdf")
    plt.show()


def plot_boxes(fig, axs_list, color):
    bboxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in axs_list]
    bboxes_fig = [fig.transFigure.inverted().transform(bbox) for bbox in bboxes]

    # Calculate the enclosing box for all the subplots in rows 2-4
    x0 = min(bbox[0, 0] for bbox in bboxes_fig)
    y0 = min(bbox[0, 1] for bbox in bboxes_fig)
    x1 = max(bbox[1, 0] for bbox in bboxes_fig)
    y1 = max(bbox[1, 1] for bbox in bboxes_fig)

    width = x1 - x0
    height = y1 - y0

    # Add a box around the 3-row area
    rect2 = Rectangle((x0, y0), width, height,
                      linewidth=3, edgecolor=color, facecolor='none', linestyle='-')
    fig.add_artist(rect2)


def delete_edges_with_same_color(graph, node_to_label):
    """
    Remove edges where both nodes have the same label.

    :param graph: The input graph (networkx.Graph)
    :param node_to_label: Dictionary mapping nodes to their labels
    :return: Modified graph with edges removed where nodes have the same label
    """
    edges_to_remove = [(u, v) for u, v in graph.edges() if node_to_label[u] == node_to_label[v]]
    graph.remove_edges_from(edges_to_remove)
    return graph


def read_cliques_from_file(filename):
    """
    Read a list of cliques from a file.

    :param filename: Path to the file containing cliques
    :return: A list of cliques (each clique is a list of nodes)
    """
    list_of_cliques = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                list_of_cliques.append(ast.literal_eval(line))  # Convert string representation to list
    return list_of_cliques


def find_bk_cliques(graph, node_to_label):
    """
    Find all Bron–Kerbosch cliques of maximum size in a graph.

    :param graph: The input graph
    :param node_to_label: Dictionary mapping nodes to their labels
    :return: A list of cliques of the largest size found in the graph
    """
    graph_without_same = delete_edges_with_same_color(graph, node_to_label)  # Remove same-label edges
    all_cliques = list(nx.find_cliques(graph_without_same))  # Find all cliques
    max_clique_size = max(len(clique) for clique in all_cliques)  # Find the size of the largest clique
    bk_cliques = [clique for clique in all_cliques if len(clique) == max_clique_size]  # Filter by max size
    return bk_cliques


def calculate_probabilities_in_clique(list_of_cliques, with_h=False):
    """
    Calculate probabilities based on cliques and whether the 'with_h' flag is considered.

    :param list_of_cliques: List of cliques (each clique is a list of nodes)
    :param with_h: Boolean flag indicating if 'h' should be considered
    :return: Dictionary of calculated probabilities
    """
    Q = list_of_cliques[-1]  # The final clique (Q)
    options_for_next = {}  # Store options for the next element in the clique

    for M in list_of_cliques:
        if M:
            new = len(M) - 1
            if with_h:
                # Count how many elements in M[:-1] are in Q
                in_Q_count = sum(1 for num in M[:-1] if num in Q)
                # Create a tuple (count in Q, count not in Q)
                options_key = (in_Q_count, len(M[:-1]) - in_Q_count)
            else:
                if set(M[:-1]) <= set(Q):
                    options_key = new
                else:
                    continue
            options_for_next.setdefault(options_key, []).append(M[-1])

    probabilities = {}
    for key, values in options_for_next.items():
        if values:
            # Calculate probability based on 'with_h' flag
            if with_h:
                probabilities[key] = any(item in Q for item in values) / len(set(values))
            else:
                probabilities[key] = 1 / len(set(values))
        else:
            probabilities[key] = 0  # No values lead to a probability of 0
    return probabilities


def aggregate_probabilities(all_probs):
    """
    Aggregate probabilities across multiple runs, calculate mean and standard deviation.

    :param all_probs: Dictionary containing probability data from multiple runs
    :return: Aggregated probabilities with mean and standard deviation for each key
    """
    aggregated_probs = {}
    for k, probs in all_probs.items():
        keys_merge = {key for prob in probs for key in prob.keys()}
        all_values = {key: [] for key in keys_merge}

        # Accumulate values across different probability dictionaries
        for prob_dict in probs:
            for key, value in prob_dict.items():
                all_values[key].append(value)
        for key in keys_merge:
            if key not in aggregated_probs:
                aggregated_probs[key] = {}
        # Calculate mean and standard deviation for each key
        for key, values in all_values.items():
            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))
            aggregated_probs.setdefault(key, {})[k] = (mean, std)

    return aggregated_probs


def data_for_plot_prob_next(num_runs, gate_or_heuristic, option="gate", p=0.3, num_k=3, first_k=9, colored=[],
                            non_colored=[], with_h=False):
    """
    Generate data for plotting probability values over multiple runs.

    :param num_runs: Number of runs to perform for each k
    :param gate_or_heuristic: Boolean flag to control clique finding process.
    :param option: The option processed by, gate or heuristic.
    :param p: Probability parameter used in graph generation
    :param num_k: Number of different values of k to test
    :param first_k: Starting value of k
    :param colored: List of colored graphs for that option.
    :param non_colored: List of non-colored graphs for that option.
    :param with_h: Boolean flag to consider h during probability calculation
    :return: Aggregated probability data for plotting
    """
    all_probs = {}
    if option == "gate":
        search_range = range(first_k, first_k + num_k)
    else:
        search_range = colored + non_colored
    for k in search_range:
        i = 0
        while i < num_runs:
            if option == "gate":
                process_graph.generate_gnp(k, p, nodes_per_color=100)  # Generate random graph
                file_name = f"erdos_renyi_graph_p={p}_{k}_classes"  # Construct file name
                with open(file_name, 'rb') as f:
                    graph, node_to_label, label_to_node = pickle.load(f)  # Load graph and node-label dictionaries
            else:
                graph_name = "../real_graphs/" + k
                if k in colored:
                    # Load and process a real colored graph
                    graph = process_graph.create_real_graph(graph_name + ".edges")
                    graph, node_to_label = process_graph.labeled_graph(graph, graph_name + ".node_labels")
                else:
                    # Load and process a real non-colored graph
                    graph = process_graph.create_real_graph(graph_name)
                    node_to_label = process_graph.color_nodes(graph)
                label_to_node = process_graph.create_label_dict(node_to_label)
                graph = process_graph.plant_clique(graph, label_to_node)
            if option == "gate":
                find_rainbow_clique.sphera(graph, node_to_label, label_to_node, False, gate_or_heuristic)
            else:
                find_rainbow_clique.sphera(graph, node_to_label, label_to_node, gate_or_heuristic, True,
                                           name=k)
            filename = f'all_way_clique_{k}.txt'  # File containing cliques
            list_of_cliques = read_cliques_from_file(filename)

            # Clear the file contents to reuse the file
            with open(filename, 'w'):
                pass  # Truncate the file, leaving it empty
            if option == "gate":
                bk_cliques = find_bk_cliques(graph, node_to_label)  # Find Bron-Kerbosch cliques
                if len(bk_cliques) == 1:
                    i += 1
                    probs = calculate_probabilities_in_clique(list_of_cliques, with_h)
                    # Save probabilities for this k
                    if k in all_probs:
                        all_probs[k].append(probs)
                    else:
                        all_probs[k] = [probs]
            else:
                i += 1
                probs = calculate_probabilities_in_clique(list_of_cliques, with_h)
                # Save probabilities for this k
                if k in all_probs:
                    all_probs[k].append(probs)
                else:
                    all_probs[k] = [probs]

    # Save all_probs to a pickle file
    if option == "gate":
        with open(f'all_probs_{p}_h={with_h}_gate={gate_or_heuristic}_200_2.pkl', 'wb') as f:
            pickle.dump(all_probs, f)
    else:
        with open(f'../results_real_graphs/all_probs_h={with_h}_heuristic={gate_or_heuristic}.pkl', 'wb') as f:
            pickle.dump(all_probs, f)
    # Calculate aggregated probabilities (mean/std) across all k values
    aggregated_probs = aggregate_probabilities(all_probs)
    return aggregated_probs


def plot_function_prob(p, t_min, t_max, axs, gate=True):
    """
    Plot theoretical and data-driven probabilities for a given range of t values.

    :param p: Probability parameter used in the theoretical function
    :param t_min: Minimum value of t to plot
    :param t_max: Maximum value of t to plot
    :param axs: The axes for plotting.
    :param gate: Boolean flag to control the clique finding process
    """

    # Helper function to extract values for a given k
    def extract_values(data_dict, k):
        x_values = list(data_dict.keys())
        y_values = [v.get(k, (None, None))[0] for v in data_dict.values()]  # Means
        y_std = [v.get(k, (None, None))[1] for v in data_dict.values()]  # Standard deviations

        # Filter out None values
        filtered_x = [x for x, y in zip(x_values, y_values) if y is not None]
        filtered_y = [y for y in y_values if y is not None]
        filtered_std = [std for std in y_std if std is not None]
        return filtered_x, filtered_y, filtered_std
    # aggregated_probs = data_for_plot_prob_next(300, gate, "gate", p, num_k=3, first_k=9, with_h=False)
    # Uncomment the following line to load pre-saved probabilities from a pickle file
    with open(f'all_probs_{p}_h=False_gate={gate}.pkl', 'rb') as f:
        all_probs = pickle.load(f)
    # Calculate the probabilities (mean/std) across all k values
    aggregated_probs = aggregate_probabilities(all_probs)
    # Extract and plot data for k = 9, 10, 11
    x_9, y_9, std_9 = extract_values(aggregated_probs, 9)
    x_10, y_10, std_10 = extract_values(aggregated_probs, 10)
    x_11, y_11, std_11 = extract_values(aggregated_probs, 11)

    # Plotting
    t_values = np.arange(t_min, t_max + 1)  # Generate t values from t_min to t_max
    y_values_theoretical = 1 / (1 + 99 * np.power(p, t_values))  # Calculate theoretical values
    # Plot the theoretical probability values for comparison
    axs.plot(t_values, y_values_theoretical, marker='o', linestyle='-', color='blue', label='Theory')

    # Plot for k=9, k=10, and k=11
    def plot_k_data(x, y, std, color, label):
        """
        Helper function to plot data points with standard deviation as error bars.

        :param x: x values (t-values)
        :param y: y values (probabilities)
        :param std: standard deviation values for error bars
        :param color: color for the plot
        :param label: label for the plot legend
        """
        # num_graphs = 300
        # std = np.array(std) / np.sqrt(num_graphs)

        axs.plot(x, y, marker='o', linestyle='-', color=color, label=label)  # Plot data points with specified color
        axs.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std), color=color, alpha=0.2)

    plot_k_data(x_9, y_9, std_9, 'red', 'k=9')
    plot_k_data(x_10, y_10, std_10, 'green', 'k=10')
    plot_k_data(x_11, y_11, std_11, 'brown', 'k=11')
    title = "With K-core" if gate else "Without K-core"
    axs.set_title(title, fontsize=font_size, fontproperties=font, loc='center')
    axs.set_xlabel('t', fontsize=font_size, fontproperties=font)
    axs.set_ylabel('Probability', fontsize=font_size, fontproperties=font) if not gate else None
    # Apply the font properties to X and Y tick labels
    for label in axs.get_xticklabels() + axs.get_yticklabels():
        label.set_fontproperties(font)
    if gate:
        axs.set_yticklabels([])  # Remove the y-axis labels
    axs.grid()
    legend_font = FontProperties()
    legend_font.set_family('serif')
    legend_font.set_name('Times New Roman')
    font_size_legend = 16
    legend_font.set_size(font_size_legend)
    axs.legend(prop=legend_font, loc='lower right') if gate else None



def find_most_common(data_dict, top_n=3):
    """
    Function to find the most common values for 't' and 'h' in a data dictionary.

    :param data_dict: Dictionary where keys are tuples (t, h), and values are nested dictionaries
    :param top_n: The number of top frequent values to return for both t and h (default is 3)
    :return: Two lists, one for the most common t values and another for the most common h values
    """
    t_counter = Counter()  # Counter to track occurrences of 't' values
    h_counter = Counter()  # Counter to track occurrences of 'h' values
    # Loop through each key (t, h) in the dictionary and count occurrences
    for (t, h) in data_dict.keys():
        t_counter[t] += 1  # Increment count for t
        h_counter[h] += 1  # Increment count for h

    # Retrieve the top N most common values for t and h
    most_common_t = t_counter.most_common(top_n)
    most_common_h = h_counter.most_common(top_n)
    # Extract just the t and h values from the tuples (t, count) for the most common
    t_values = [t for t, _ in most_common_t]
    h_values = [h for h, _ in most_common_h]

    return t_values, h_values


def plot_subplot_with_h(data_dict, fixed_value, var_values, fixed_label, var_label, p, ax, colors, range_k, index, gate,
                        row):
    """
    Function to plot a subplot with error bars and the theoretical curve for fixed and varying parameters (t or h).

    :param data_dict: Dictionary containing data to plot
    :param fixed_value: The fixed value for either t or h, depending on the plot
    :param var_values: The varying values for either t or h, depending on the plot
    :param fixed_label: Label for the fixed parameter ('t' or 'h')
    :param var_label: Label for the variable parameter ('t' or 'h')
    :param p: Probability parameter used for the theoretical curve
    :param ax: The axes on which to plot
    :param colors: Color mapping for different k values
    :param range_k: The range of k values to plot
    :param index: The index of the subplot.
    :param gate: Bool for gate.
    :param row: Row of subplot.
    """
    # Loop through each k value in the specified range
    for k in range_k:
        # var_points, means, stds = [], [], []  # Lists to hold variable values, means, and standard deviations
        points, means, stds = [], [], []  # Lists to hold variable values, means, and standard deviations

        # Loop through the dictionary to extract the data for the fixed parameter and varying parameter
        for (t, h), k_dict in data_dict.items():
            fixed_key = t if fixed_label == 't' else h
            var_key = h if fixed_label == 't' else t

            # Check if the fixed parameter matches the given fixed value and if k exists in the data
            if fixed_key == fixed_value and k in k_dict:
                mean, std = k_dict[k]
                points.append(var_key + fixed_key)  # add n
                means.append(mean)
                stds.append(std)

        # Sort the points based on the varying parameter (t or h)
        sorted_indices = np.argsort(points)
        points = np.array(points)[sorted_indices]
        means = np.array(means)[sorted_indices]
        stds = np.array(stds)[sorted_indices]

        # Helper function to plot the data with error bars
        def plot_with_error_bars(ax, x_points, y_means, y_stds, color, label):
            """
            Helper function to plot data with error bars and the shaded area for standard deviation.

            :param ax: The axes to plot on
            :param x_points: x values (t or h)
            :param y_means: Mean values to plot
            :param y_stds: Standard deviations for error bars
            :param color: Color for the plot
            :param label: Label for the plot legend
            """
            if y_means.size > 0:
                ax.plot(x_points, y_means, marker='o', linestyle='-', color=color, label=label)
                ax.fill_between(x_points, y_means - y_stds, y_means + y_stds, color=color, alpha=0.2)

        # Plot the data for the current k value with error bars
        plot_with_error_bars(ax, points, means, stds, color=colors[k], label=f'k={k}')

    # Plot the theoretical curve: (p^h) / (1 + 99 * p^n)
    if fixed_label == "t":
        theoretical_y = np.power(p, var_values) / (1 + 99 * np.power(p, fixed_value + var_values))
    else:
        theoretical_y = np.power(p, fixed_value) / (1 + 99 * np.power(p, var_values + fixed_value))

    # Plot the theoretical curve on the same axis
    ax.plot(var_values, theoretical_y, marker='o', color='blue', label='Theory', linestyle='-')

    # Set title and labels for the plot
    equation = f'{fixed_label} = {fixed_value}' if fixed_label == "h" else f'n - h = {fixed_value}'
    title = equation
    ax.set_title(title, fontsize=font_size, fontproperties=font)
    ax.set_ylabel('Probability', fontsize=font_size, fontproperties=font) if not gate else None
    ax.set_xlim(var_values.min(), var_values.max())
    if row == 1:
        if index == 1:
            ax.set_ylim(0, 0.2)
        elif index == 0:
            ax.set_ylim(0, 0.1)
    else:
        ax.set_ylim(-0.05, 1.05)
    # Apply the font properties to X and Y tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    ax.grid()
    if row != 4 or index != 1:
        ax.set_xticklabels([])  # Remove the x-axis labels
    if gate:
        ax.set_yticklabels([])  # Remove the y-axis labels
    var_label = var_label if var_label == "h" else "n"
    if (index == 2 and row == 1) or (index == 1 and row == 4):
        ax.set_xlabel(var_label, fontsize=font_size, fontproperties=font)



def plot_function_prob_with_h(p, t_min, t_max, h_min, h_max, axs, gate=True):
    """
    Function to plot the probability distribution by varying t and h values.

    :param p: Probability parameter used for theoretical curve
    :param t_min: Minimum value for t
    :param t_max: Maximum value for t
    :param h_min: Minimum value for h
    :param h_max: Maximum value for h
    :param axs: The axes for plotting.
    :param gate: Boolean flag indicating whether to include the gate parameter
    """
    # Generate arrays for t and h values within the specified range
    t_values = np.arange(t_min, t_max + 1)
    h_values = np.arange(h_min, h_max + 1)
    first_k = 9  # Initial value for k
    # Generate data for the plot
    #data_dict = data_for_plot_prob_next(400, gate, "gate", p, num_k=3, first_k=first_k, with_h=True)
    # Uncomment the following line to load pre-saved probabilities from a pickle file
    with open(f'all_probs_{p}_h=True_gate={gate}_combined_1400.pkl', 'rb') as f:
        all_probs = pickle.load(f)
    data_dict = aggregate_probabilities(all_probs)  # Aggregate probabilities if using the pickle file
    print(data_dict)

    # Find the most common t and h values from the data
    most_common_t, most_common_h = find_most_common(data_dict)

    # Define the range of k values to plot
    range_k = [first_k, first_k + 1, first_k + 2]
    # Define colors for different k values
    colors = {first_k: 'green', first_k + 1: 'red', first_k + 2: 'brown'}

    # Plot the subplots for each most common t value
    for i, t_const in enumerate(most_common_t):
        plot_subplot_with_h(data_dict, t_const, h_values, 't', 'h', p, axs[i], colors, range_k, i, gate, 1)

    # Plot the subplots for each most common h value
    most_common_h = most_common_h[1:]
    for i, h_const in enumerate(most_common_h):
        plot_subplot_with_h(data_dict, h_const, t_values, 'h', 't', p, axs[i + 3], colors, range_k, i, gate, 4)


def combine_data(with_gate):
    with open(f'all_probs_0.3_h=True_gate={with_gate}_combined_1200.pkl', 'rb') as f:
        probs_run1 = pickle.load(f)
    with open(f'all_probs_0.3_h=True_gate={with_gate}_200_2.pkl', 'rb') as f:
        probs_run2 = pickle.load(f)

    # Initialize the merged dictionary
    combined_dict = {}

    for key in probs_run1.keys() | probs_run2.keys():  # Ensure we get all keys from both dicts
        list1 = probs_run1.get(key, [])  # Get list from dict1 or an empty list if key not found
        list2 = probs_run2.get(key, [])  # Get list from dict2 or an empty list if key not found
        combined_dict[key] = list1 + list2  # Append lists from both dictionaries

    # Resulting merged_dict will contain keys with 800 items in each list
    with open(f'all_probs_0.3_h=True_gate={with_gate}_combined_1400.pkl', 'wb') as f:
        pickle.dump(combined_dict, f)
    return combined_dict


if __name__ == '__main__':
    combine(0.3, 0, 10, 0, 10)
    # data_for_plot_prob_next(200, True, "gate", 0.3, num_k=3, first_k=9, with_h=True)
    # data_for_plot_prob_next(200, False, "gate", 0.3, num_k=3, first_k=9, with_h=True)
    # combine_data(True)
    # combine_data(False)


