import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from plots import greedy_percentage, real_graph_plots

# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font_size = 20
font.set_size(font_size)


def calculate_k_max(num_nodes, p):
    """
    Calculate the threshold of natural cliques (K_max) for a given graph.

    This function computes the maximum value of K for a graph where
    the number of nodes and edge probability is provided. The calculation
    is based on the formula derived from the Erdős–Rényi model.

    :param num_nodes: Number of nodes in the graph.
    :param p: Probability of existence for each edge in the graph.
    :return: max_k - The threshold of natural cliques (K_max).
    """
    # Compute the K_max threshold for natural cliques
    return 2 * (np.log(num_nodes) / np.log(1 / p)) - 2 * (np.log(np.log(num_nodes) / np.log(1 / p)) / np.log(1 / p))


def rainbow_clique(num_colors, nodes_per_color, p):
    """
    Calculate the maximum size of a rainbow clique in a graph.

    A rainbow clique is defined as a clique where each node comes from
    a different color. This function calculates the threshold of natural
    rainbow cliques based on the number of colors, nodes per color,
    and edge probability.

    :param num_colors: The number of different colors (sets) in the graph.
    :param nodes_per_color: Number of nodes in each color set.
    :param p: Probability of existence for each edge in the graph.
    :return: max_k - The threshold of natural rainbow cliques.
    """
    max_k = 0

    def expression(num_colors, k):
        """
        Helper function to compute the expression for the rainbow clique threshold.

        :param num_colors: The number of different colors.
        :param k: The size of the rainbow clique.
        :return: Value of the expression for given num_colors and k.
        """
        return math.comb(num_colors, k) * (nodes_per_color ** k) * (p ** ((k * (k - 1)) / 2))

    # Iterate over all possible k values and compute the expression
    for i in range(num_colors + 1):
        z = expression(num_colors, i)
        if z >= 1:
            max_k = i  # Update max_k if the current value satisfies the condition
    return max_k


def plot_for_p(p, range_k, place):
    """
    Plot the maximum clique and rainbow clique thresholds as functions of K for a given edge probability p.

    This function generates a plot for the max clique and max rainbow clique, comparing them over
    a range of K values.

    :param p: The edge probability for the graph.
    :param range_k: The range of K values to consider for the plot.
    :param place: The subplot axis object where the plot should be drawn.
    """
    # Determine the number of nodes per color based on the probability p
    if p == 0.5:
        nodes_per_color = 100
    else:
        nodes_per_color = 1000

    # Generate an array of K values to plot
    k_values = np.arange(1, range_k)

    # Calculate the max clique threshold for each K
    k_max_values = calculate_k_max(k_values * nodes_per_color, p)

    # Calculate the max rainbow clique threshold for each K
    k_max_rainbow = []
    for k in range(1, range_k):
        k_max_rainbow.append(rainbow_clique(k, nodes_per_color, p))

    # Plot the results
    place.plot(k_values, k_max_values, label="Max clique", color='red')
    place.plot(k_values, k_values, label="X=Y", color='blue')
    place.plot(k_values, k_max_rainbow, label="Max rainbow clique", color='green')
    for i, value in enumerate(k_max_values):
        if i == round(value):
            first_match_values = i
            break
    for i, value in enumerate(k_max_rainbow):
        if i == round(value):
            first_match_rainbow = i
            break

    # Set plot labels, title, and grid
    # place.set_xlabel('K \n (a)' if p == 0.3 else 'K', fontproperties=font)
    place.set_ylabel(f'K Max', fontproperties=font)  # if p == 0.1 else None
    place.set_xlabel(f"Number of colors", fontproperties=font) if p == 0.5 else None
    legend_font = FontProperties()
    legend_font.set_family('serif')
    legend_font.set_name('Times New Roman')
    font_size_legend = 16
    legend_font.set_size(font_size_legend)
    place.legend(prop=legend_font, loc='upper left') if p == 0.1 else None
    # place.set_xticks(np.arange(2, range_k, 2))
    # if p != 0.5:
    place.set_xticklabels([])
    # place.set_xticklabels(place.get_xticks(), fontproperties=font)  # Set font for X-axis tick labels
    place.set_yticklabels(place.get_yticks(), fontproperties=font)
    place.set_yticks(np.arange(2, range_k, 5))
    ytick_labels = place.get_yticks()
    place.set_yticklabels(ytick_labels, fontproperties=font)
    place.set_title(f'p = {p}', fontproperties=font)
    place.set_xlim(0, range_k)
    place.set_ylim(0, range_k)
    place.grid(True)
    threshold = min(first_match_rainbow, first_match_values)
    place.axvline(x=threshold, color='black', linestyle='--')
    return threshold



def main():
    """
    Main function to generate and display plots for different values of p (probability).

    This function creates subplots showing the relationship between K and K_max
    for both the maximum clique and rainbow clique for various edge probabilities.
    """
    # Create a 1x3 subplot grid
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)] + [fig.add_subplot(gs[i, 1]) for i in range(3)]
    # Generate and plot for different values of p (0.1, 0.3, 0.5)
    greedy_percentage.plot_gnp_results(axs[3:6])
    threshold1 = plot_for_p(0.1, 50, axs[0])
    threshold2 = plot_for_p(0.3, 50, axs[1])
    threshold3 = plot_for_p(0.5, 50, axs[2])
    threshold = [threshold1, threshold2, threshold3]
    # Determine global limits
    x_ranges_1 = (3, 21)
    x_ranges_2 = (3, 31)
    # Apply global limits to all subplots
    for i in range(6):
        x_range = x_ranges_1 if i == 0 or i == 3 else x_ranges_2
        axs[i].set_xlim(x_range)
        if i < 3:
            axs[i].set_ylim(x_range)
            axs[i].set_xticks(range(5, x_range[1] + 1, 5))
            axs[i].set_xticklabels(axs[i].get_xticks(), fontproperties=font)
            axs[i].set_yticks(range(5, x_range[1] + 1, 5))
            axs[i].set_yticklabels(axs[i].get_yticks(), fontproperties=font)
    for i in [3, 4, 5]:
        axs[i].axvline(x=threshold[i - 3], color='black', linestyle='--')

    all_cliques, _, graph_names_short = real_graph_plots.time_and_percentage_data()
    print(all_cliques)
    axs_last_row = fig.add_subplot(gs[3, :])  # Last row, spanning both columns
    real_graph_plots.subplot_boxplot(axs_last_row, all_cliques, graph_names_short, "left")
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig("Fig3.pdf", format="pdf")
    plt.show()


if __name__ == '__main__':
    main()

