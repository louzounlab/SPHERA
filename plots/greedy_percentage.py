import time
import matplotlib.pyplot as plt
import numpy as np
import sphera.find_rainbow_clique as find_rainbow_clique
from matplotlib.font_manager import FontProperties
import sphera.process_graph as process_graph
import pickle
import glob

# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font_size = 20
font.set_size(font_size)


def run_experiment(ks, num_runs, p=None, mode="gnp"):
    """
    Runs experiments to find cliques in generated or real_graphs and records execution time and clique size.

    :param ks: List of values for the number of colors or names of graphs.
    :param num_runs: Number of runs for each experiment.
    :param p: Probability parameter for G(n,p) graph generation. Default is None.
    :param mode: Mode specifying graph type ("gnp", "colored", "non-colored"). Default is "gnp".
    """
    # Define the three gate options
    gate_options = ["greedy", "sphera", "no_gate", "bk"]
    for option in gate_options:
        for k in ks:
            print(k)
            times = []
            clique_sizes = []

            for _ in range(num_runs):
                if mode == "gnp":
                    # Generate a G(n, p) graph
                    process_graph.generate_gnp(k, p, 100 if p == 0.5 else 1000)
                    # Construct the file name based on the provided parameters
                    file_name = f"erdos_renyi_graph_p={p}_{k}_classes"
                    # Load the generated graph and dictionaries.
                    with open(file_name, 'rb') as f:
                        graph, node_to_label, label_to_node = pickle.load(f)
                elif mode == "colored":
                    # Load and process a real colored graph
                    graph_name = "real_graphs/" + k
                    graph = process_graph.create_real_graph("../" + graph_name + ".edges")
                    graph, node_to_label = process_graph.labeled_graph(graph, "../" + graph_name + ".node_labels")
                    label_to_node = process_graph.create_label_dict(node_to_label)
                    graph = process_graph.plant_clique(graph, label_to_node)
                else:
                    # Load and process a real non-colored graph
                    graph_name = "../real_graphs/" + k
                    graph = process_graph.create_real_graph(graph_name)
                    node_to_label = process_graph.color_nodes(graph)
                    label_to_node = process_graph.create_label_dict(node_to_label)
                    graph = process_graph.plant_clique(graph, label_to_node)
                # Measure execution time for each option
                start_time = time.time()
                if option == "greedy":
                    clique_founded, _ = find_rainbow_clique.rc_detection(graph, node_to_label, label_to_node, heuristic=False,
                                                                   greedy=True)
                    clique_size = len(clique_founded)
                elif option == "sphera":
                    clique_founded, _ = find_rainbow_clique.rc_detection(graph, node_to_label, label_to_node, heuristic=False,
                                                                   greedy=False)
                    clique_size = len(clique_founded)
                elif option == "bk":
                    found_cliques = find_rainbow_clique.bron_kerbosch(graph.copy(), label_to_node.keys(), node_to_label,
                                                                      label_to_node, [], list(graph.nodes()),
                                                                      [], [])
                    clique_size = len(found_cliques[-1])
                elif option == "no_gate":
                    clique_founded, _ = find_rainbow_clique.rc_detection(graph, node_to_label, label_to_node, heuristic=False,
                                                                   greedy=False, gate_change=False)
                    clique_size = len(clique_founded)
                else:
                    clique_founded, _ = find_rainbow_clique.rc_detection(graph, node_to_label, label_to_node, heuristic=True,
                                                                   greedy=False)
                    clique_size = len(clique_founded)
                end_time = time.time()
                elapsed_time = end_time - start_time

                times.append(elapsed_time)
                clique_sizes.append(clique_size)

            # Save results to a separate file for each (p, k, option) combination
            if mode == "gnp":
                filename = f"results_p={p}_k={k}_{option}.txt"
            else:
                filename = f"../results_real_graphs/results_k={k}_{option}_fixed.txt"
            with open(filename, 'w') as f:
                for t, s in zip(times, clique_sizes):
                    f.write(f"{t},{s}\n")


def plot_gnp_results(axs):
    """
    Plots the results for the G(n,p) graphs showing the average and standard deviation
    of clique sizes across different numbers of colors (k) for varying probabilities (p).
    """
    # Define values of p and initialize a dictionary to hold the data for each p value
    p_values = [0.1, 0.3, 0.5]
    all_k_values = {p: set() for p in p_values}
    methods = ['greedy', 'sphera', 'heuristic', 'no_gate']
    title_mapping = {'greedy': 'Greedy', 'no_gate': 'SP', 'sphera': 'SP (k-Core)', 'heuristic': 'SP (Heu)'}
    data = {p: {method: {} for method in methods} for p in p_values}
    # Organize data into a dictionary from the result files
    for p in p_values:
        num_graphs = 20
        for method in methods:
            # Get all files for the current p and method
            files = glob.glob(f"results_p={p}_k=*_{method}.txt")

            for file in files:
                # Extract k from the filename
                k = int(file.split("_k=")[1].split("_")[0])
                all_k_values[p].add(k)
                with open(file, 'r') as f:
                    probs = [int(line.split(",")[1].strip()) / k for line in f.readlines()]
                    percentages = [int(prob * 100) for prob in probs]
                    # class_values = [int(line.split(",")[1].strip()) for line in f.readlines()]
                    # Calculate mean and standard deviation for each k
                    avg = np.mean(percentages)
                    std = np.std(percentages) / np.sqrt(num_graphs)
                    data[p][method][k] = (avg, std)

    # Create subplots for each value of p
    #fig, axes = plt.subplots(1, len(p_values), figsize=(15, 5), sharey=True)
    colors = ['red', 'green', 'blue', 'brown']
    for i, p in enumerate(p_values):
        ax = axs[i]
        ax.set_xlabel(f"Number of colors", fontproperties=font) if p == 0.5 else None
        #if i == 0:
        ax.set_ylabel("Clique size detected (%)", fontproperties=font)
        if p != 0.5:
            ax.set_xticklabels([])

        for j, method in enumerate(methods):
            # Extract k values, averages, and standard deviations
            k_values = sorted(data[p][method].keys())
            averages = [data[p][method][k][0] for k in k_values]
            std_devs = [data[p][method][k][1] for k in k_values]

            # Plot with error bars
            ax.errorbar(k_values, averages, yerr=std_devs, label=title_mapping[method], color=colors[j])

        x_values = sorted(data[p]['greedy'].keys())
        x_ticks = np.arange(0, max(x_values) + 1, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(ax.get_xticks(), fontproperties=font)
        ax.set_yticklabels(ax.get_yticks(), fontproperties=font)
        legend_font = FontProperties()
        legend_font.set_family('serif')
        legend_font.set_name('Times New Roman')
        font_size_legend = 16
        legend_font.set_size(font_size_legend)
        ax.legend(prop=legend_font, loc='upper right') if p == 0.1 else None
        ax.grid(True)
        ax.set_title(f'p = {p}', fontproperties=font)

    plt.tight_layout()


def run_options_for_plots():
    """
    Runs experiments for both G(n,p) and real-world graphs and saves the results to files.
    """
    p_values = [0.3]
    for i, p in enumerate(p_values):
        if p == 0.1:
            ks = range(3, 21)
        elif p == 0.3:
            ks = range(30, 31)
        else:
            ks = range(3, 31)
        run_experiment(ks, 20, p)
    colored_graphs = ["DHFR-MD", "AIDS", "DD242", "COX2", "soc-Flickr-ASU"]
    run_experiment(colored_graphs, 10, mode="colored")

    non_colored_graphs = ["artist_edges.csv", "CA-CondMat.txt", "Email-Enron.txt", "large_twitch_edges.csv",
                          "musae_git_edges.csv", "oregon1_010428.txt"]
    run_experiment(non_colored_graphs, 10, mode="non colored")


if __name__ == '__main__':
    run_options_for_plots()
    # plot_gnp_results()
    # plot_real_graph_results()

