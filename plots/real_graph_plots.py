import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties

# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font_size = 20
font.set_size(font_size)


def time_and_percentage_data():
    """
    Collects and processes results from different methods applied to real graphs.
    The function compares SPHERA (with and without heuristics) and the Bron-Kerbosh algorithm
    by reading their results, calculating clique size percentages, and gathering time data.
    Returns:
        all_cliques (dict): Dictionary containing clique size percentages for each method.
        all_times (dict): Dictionary containing computation times for each method.
        graph_names_short (list): List of simplified graph names for visualization.
    """
    # Define the dictionary with graph sizes
    clique_sizes = {
        "DHFR-MD": 7,
        "AIDS": 38,
        "DD242": 20,
        "COX2": 8,
        "soc-Flickr-ASU": 195,
        "artist_edges.csv": 32,
        "CA-CondMat.txt": 8,
        "Email-Enron.txt": 10,
        "large_twitch_edges.csv": 15,
        "musae_git_edges.csv": 15,
        "oregon1_010428.txt": 4
    }

    # Define methods and files
    methods = ["bk", "greedy", "no_gate", "sphera", "heuristic"]

    # Initialize lists to store results for plotting
    graph_names = []
    all_times = {method: [] for method in methods}
    all_cliques = {method: [] for method in methods}

    # Process each graph
    for graph, size in clique_sizes.items():
        graph_names.append(graph)  # Store graph names for plotting
        for method in methods:
            filename = f"../results_real_graphs/results_k={graph}_{method}_fixed.txt"

            # Load the file and calculate mean/std for time and clique size
            if os.path.exists(filename):
                data = pd.read_csv(filename, header=None)
                times = data[0]
                cliques = data[1]
                clique_percentage = (cliques / size) * 100
                all_cliques[method].append(list(clique_percentage))
                all_times[method].append(list(times))

            else:
                print(f"File {filename} not found.")
    graph_names_short = [name.split('.')[0] for name in graph_names]
    return all_cliques, all_times, graph_names_short


def subplot_boxplot(axes, method_data, graph_names_short, position):
    """
    Creates boxplots with scatter overlays to compare method performance.
    Args:
        axes (matplotlib.axes.Axes): Axes to plot on.
        method_data (dict): Dictionary containing data for each method.
        graph_names_short (list): Simplified graph names for labeling.
        position (str): Indicates the type of data ('left' for clique size, 'right' for time).
    """
    # Prepare the data for plotting
    method_means, method_graph_names, method_names = [], [], []
    # Calculate mean and std for each method and graph
    for method, graphs in method_data.items():
        for i, graph_runs in enumerate(graphs):
            method_means.extend(graph_runs)
            method_graph_names.extend([graph_names_short[i]] * len(graph_runs))  # Replicate the graph name for each run
            method_names.extend([method] * len(graph_runs))  # Replicate the method name for each run

    # Create DataFrame from the prepared data
    df = pd.DataFrame({
        'Method': method_names,
        'Graph': method_graph_names,
        'Value': method_means
    })

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'teal', 'navy', 'gray', 'hotpink']

    # Add the dots for each individual run
    legend_font = FontProperties()
    legend_font.set_family('serif')
    legend_font.set_name('Times New Roman')
    font_size_legend = 16
    legend_font.set_size(font_size_legend)
    sns.stripplot(x='Method', y='Value', data=df, hue='Graph', palette=colors, dodge=True, marker='o', alpha=0.7,
                  jitter=0.3, size=6, ax=axes, legend=True)
    # Add vertical lines between the methods
    num_methods = len(df['Method'].unique())
    for i in range(1, num_methods):
        axes.axvline(x=i - 0.5, color='black', linewidth=1)
    # Adjust the legend to be outside the plot
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles, labels=graph_names_short, title='Graphs', loc='upper left', bbox_to_anchor=(1, 1),
                prop=legend_font)

    # Customize labels
    axes.set_xlabel('Method', fontproperties=font)
    if position == "left":
        axes.set_ylabel('Clique size detected (%)', fontproperties=font)
    else:
        axes.set_ylabel('Time (seconds)', fontproperties=font)
        axes.set_yscale('log')
    # Define the mapping {current_title: new_title}
    title_mapping = {'bk': 'BK', 'greedy': 'Greedy', 'no_gate': 'SP', 'sphera': 'SP (K-core)', 'heuristic': 'SP (Heu)'}

    # Get the current tick labels
    current_titles = axes.get_xticklabels()
    current_labels = [tick.get_text() for tick in current_titles]

    # Update the labels using the dictionary
    new_labels = [title_mapping.get(label, label) for label in current_labels]

    # Set the new x-axis tick labels
    axes.set_xticklabels(new_labels)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, fontproperties=font)
    axes.set_yticklabels(axes.get_yticklabels(), fontproperties=font)
    axes.grid(visible=True, which='both', axis='both', linewidth=0.5, alpha=0.7)


if __name__ == '__main__':

    _, all_times, graph_names_short = time_and_percentage_data()
    fig, ax = plt.subplots(figsize=(14, 6))
    subplot_boxplot(ax, all_times, graph_names_short, "right")  # Pass ax directly
    plt.tight_layout()
    plt.savefig("Fig5.pdf", format="pdf")
    plt.show()



