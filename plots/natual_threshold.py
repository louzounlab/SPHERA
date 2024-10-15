import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Setting font properties for consistent styling in the plot
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')


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
    place.plot(k_values, k_max_values, label="max clique")
    place.plot(k_values, k_values)
    place.plot(k_values, k_max_rainbow, label="max rainbow clique")

    # Set plot labels, title, and grid
    place.set_xlabel('K', fontproperties=font, fontsize=20)
    place.set_ylabel(f'K_max', fontproperties=font, fontsize=20)
    place.legend()
    place.set_xticks(np.arange(2, len(k_values), 2))
    place.set_yticks(np.arange(2, len(k_values), 2))
    place.set_title(f'p = {p}', fontproperties=font, fontsize=20)
    place.set_xlim(0, 25)
    place.set_ylim(0, 25)
    place.grid(True)


def main():
    """
    Main function to generate and display plots for different values of p (probability).

    This function creates subplots showing the relationship between K and K_max
    for both the maximum clique and rainbow clique for various edge probabilities.
    """
    # Create a 1x3 subplot grid
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Generate and plot for different values of p (0.1, 0.3, 0.5)
    plot_for_p(0.1, 50, axs[0])
    plot_for_p(0.3, 50, axs[1])
    plot_for_p(0.5, 50, axs[2])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
