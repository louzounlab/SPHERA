import ast
import pickle
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import find_rainbow_clique
import process_graph
from matplotlib.font_manager import FontProperties


# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')

def delete_edges_with_same_color(graph, node_to_label):
    """Remove edges where both nodes have the same label."""
    edges_to_remove = [(u, v) for u, v in graph.edges() if node_to_label[u] == node_to_label[v]]
    graph.remove_edges_from(edges_to_remove)
    return graph


def read_cliques_from_file(filename):
    """Read a list of cliques from a file."""
    list_of_cliques = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                list_of_cliques.append(ast.literal_eval(line))
    return list_of_cliques

def find_bk_cliques(graph, node_to_label):
    """Find all Bron–Kerbosch cliques of maximum size in a graph."""
    graph_without_same = delete_edges_with_same_color(graph, node_to_label)
    all_cliques = list(nx.find_cliques(graph_without_same))
    max_clique_size = max(len(clique) for clique in all_cliques)
    bk_cliques = [clique for clique in all_cliques if len(clique) == max_clique_size]
    return bk_cliques
def calculate_probabilities_in_clique(list_of_cliques, with_h=False):
    """Calculate probabilities based on cliques and whether with_h is considered."""

    Q = list_of_cliques[-1]

    # Initialize the dictionary
    options_for_next = {}
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
    # Calculate probabilities
    probabilities = {}
    for key, values in options_for_next.items():
        if values:
            if with_h:
                probabilities[key] = any(item in Q for item in values) / len(set(values))
            else:
                probabilities[key] = 1 / len(set(values))  # Probability is 1 / length of the list
        else:
            probabilities[key] = 0  # If there are no values, probability is 0
    return probabilities


def aggregate_probabilities(all_probs):
    """Aggregate probabilities across multiple runs, calculate mean and standard deviation."""
    aggregated_probs = {}
    for k, probs in all_probs.items():
        keys_merge = {key for prob in probs for key in prob.keys()}
        all_values = {key: [] for key in keys_merge}

        for prob_dict in probs:
            for key, value in prob_dict.items():
                all_values[key].append(value)
        # ?
        for key in keys_merge:
            if key not in aggregated_probs:
                aggregated_probs[key] = {}

        for key, values in all_values.items():
            mean = np.mean(values)
            std = np.std(values)
            aggregated_probs.setdefault(key, {})[k] = (mean, std)

    return aggregated_probs

def data_for_plot_prob_next(num_runs, gate, p=0.3, num_k=3, first_k=9, with_h=False):
    all_probs = {}

    for k in range(first_k, first_k + num_k):
        i = 0
        print("a")
        while i < num_runs:
            print(i)
            process_graph.generate_gnp(k, p, nodes_per_color=100)
            # Construct the file name based on the provided parameters
            file_name = f"erdos_renyi_graph_p={p}_{k}_classes"
            # Load the generated graph and dictionaries from the .gpickle file
            with open(file_name, 'rb') as f:
                graph, node_to_label, label_to_node = pickle.load(f)
            find_rainbow_clique.sphere(graph, node_to_label, label_to_node, False, gate)
            filename = f'all_way_clique_{k}.txt'  # Change to the path of your file
            list_of_cliques = read_cliques_from_file(filename)
            # Clear the file contents (blank the file)
            with open(filename, 'w'):
                pass  # This will truncate the file, leaving it empty
            bk_cliques = find_bk_cliques(graph, node_to_label)
            if len(bk_cliques) == 1:
                i += 1
                probs = calculate_probabilities_in_clique(list_of_cliques, with_h)
                # Save probabilities for this k
                if k in all_probs:
                    all_probs[k].append(probs)
                else:
                    all_probs[k] = [probs]

    # Save all_probs to a pickle file
    with open(f'all_probs_{p}_{with_h}.pkl', 'wb') as f:
        pickle.dump(all_probs, f)
    # Calculate the probabilities (mean/std) across all k values
    aggregated_probs = aggregate_probabilities(all_probs)
    return aggregated_probs


def plot_function_prob(p, t_min, t_max, gate=True):
    """Plot theoretical and data probabilities for t_min to t_max."""
    # Generate t values
    t_values = np.arange(t_min, t_max + 1)  # +1 to include t_max in the array
    # Compute the function values
    y_values_theoretical = 1 / (1 + 99 * np.power(p, t_values))

    # Extract values for each k
    def extract_values(data_dict, k):
        x_values = list(data_dict.keys())
        y_values = [v[k][0] for v in data_dict.values()]
        y_std = [v[k][1] for v in data_dict.values()]
        filtered_x = [x for x, y in zip(x_values, y_values) if y is not None]
        filtered_y = [y for y in y_values if y is not None]
        filtered_std = [std for std in y_std if std is not None]
        return filtered_x, filtered_y, filtered_std

    aggregated_probs = data_for_plot_prob_next(2, gate, p, num_k=3, first_k=9, with_h=False)

    # Get k-specific data
    x_9, y_9, std_9 = extract_values(aggregated_probs, 9)
    x_10, y_10, std_10 = extract_values(aggregated_probs, 10)
    x_11, y_11, std_11 = extract_values(aggregated_probs, 11)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values_theoretical, marker='o', linestyle='-', color='blue', label='Theory')

    # Plot for k=9, k=10, and k=11
    def plot_k_data(x, y, std, color, label):
        plt.plot(x, y, marker='o', linestyle='-', color=color, label=label)
        plt.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std), color=color, alpha=0.2)

    plot_k_data(x_9, y_9, std_9, 'red', 'Data Points (k=9)')
    plot_k_data(x_10, y_10, std_10, 'green', 'Data Points (k=10)')
    plot_k_data(x_11, y_11, std_11, 'brown', 'Data Points (k=11)')

    plt.xlabel('t')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'without_h_gate={gate}.png')  # Save the plot as a PNG file
    plt.show()


def merge_and_find_most_common(first_dict, second_dict, k_1, k_2, top_n=3):
    # Combine the two dictionaries, keeping both k=9 and k=10 entries
    data_dict = {}
    for key in set(first_dict) | set(second_dict):  # Union of keys
        data_dict[key] = {}
        if key in first_dict:
            data_dict[key][k_1] = first_dict[key].get(k_1, (None, None))  # Keep k=9 if available
        if key in second_dict:
            data_dict[key][k_2] = second_dict[key].get(k_2, (None, None))  # Keep k=10 if available

    # Count occurrences of t and h values, including zeroes if necessary
    t_count = Counter()
    h_count = Counter()

    for (t, h), k_dict in data_dict.items():
        t_count[t] += 1
        h_count[h] += 1

    # Get the top N most common t and h values
    most_common_t = [item[0] for item in t_count.most_common(top_n)]
    most_common_h = [item[0] for item in h_count.most_common(top_n)]

    return most_common_t, most_common_h

def plot_subplot_with_h(data_dict, fixed_value, var_values, fixed_label, var_label, p, ax, colors):
    """
    General function to plot data for fixed and varying parameters (either t or h).
    """
    for k in [9, 10]:
        var_points, means, stds = [], [], []

        for (t, h), k_dict in data_dict.items():
            fixed_key = t if fixed_label == 't' else h
            var_key = h if fixed_label == 't' else t

            if fixed_key == fixed_value and k in k_dict:
                mean, std = k_dict[k]
                var_points.append(var_key)
                means.append(mean)
                stds.append(std)

        # Sort points based on the variable (t or h)
        sorted_indices = np.argsort(var_points)
        var_points = np.array(var_points)[sorted_indices]
        means = np.array(means)[sorted_indices]
        stds = np.array(stds)[sorted_indices]

        def plot_with_error_bars(ax, x_points, y_means, y_stds, color, label):
            """
            Helper function to plot error bars and connecting lines.
            """
            if y_means.size > 0:
                ax.errorbar(x_points, y_means, yerr=y_stds, fmt='o', color=color, capsize=5, label=label)
                ax.plot(x_points, y_means, color=color)
        # Plot data with error bars
        plot_with_error_bars(ax, var_points, means, stds, color=colors[k], label=f'k={k}')

    # Plot theoretical line
    theoretical_y = np.power(p, var_values) / (1 + 99 * np.power(p, fixed_value))
    ax.plot(var_values, theoretical_y, marker='o', color='blue', label='Theory', linestyle='-')

    ax.set_title(f'{fixed_label} = {fixed_value}')
    ax.set_ylabel('Mean Value')
    ax.set_xlim(var_values.min(), var_values.max())
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    ax.legend()
    ax.set_xlabel(var_label)

def plot_function_prob_with_h(p, t_min, t_max, h_min, h_max, gate=True):
    # Generate t and h values (integer values)
    t_values = np.arange(t_min, t_max + 1)
    h_values = np.arange(h_min, h_max + 1)
    first_k = 9
    data_dict = data_for_plot_prob_next(2, gate, p, num_k=2, first_k=first_k, with_h=True)
    most_common_t, most_common_h = merge_and_find_most_common(data_dict[first_k], data_dict[first_k + 1], first_k,
                                                              first_k + 1)
    print(most_common_t)
    print(most_common_h)

    # Plotting the first figure for t
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle('Mean Values by h for Most Common t Values')
    # Define colors for different k values
    colors = {first_k: 'green', first_k + 1: 'red'}

    for i, t_const in enumerate(most_common_t):
        plot_subplot_with_h(data_dict, t_const, h_values, 't', 'h', p, axs[i], colors)

    plt.tight_layout()
    plt.savefig(f'with_h_gate={gate}_var_h.png')  # Save the plot as a PNG file
    plt.show()
    # Plotting the second figure for h
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle('Mean Values by t for Most Common h Values')

    for i, h_const in enumerate(most_common_h):
        plot_subplot_with_h(data_dict, h_const, t_values, 'h', 't', p, axs[i], colors)

    plt.tight_layout()
    plt.savefig(f'with_h_gate={gate}_var_t.png')
    plt.show()


if __name__ == '__main__':
    plot_function_prob(0.3, 0, 10)
    plot_function_prob_with_h(0.3, 0, 10, 0, 10)
