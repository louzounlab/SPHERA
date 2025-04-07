import pickle
import networkx as nx
from itertools import combinations
import random
import re


def create_label_dict(node_to_label):
    """
    Returns a dictionary that maps each label to the corresponding list of nodes.

    :param node_to_label: Dictionary mapping nodes to labels
    :return: Dictionary mapping labels to lists of nodes associated with that label
    """
    labels_set = set(node_to_label.values())  # Extract unique labels from node_to_label
    label_to_node = {key: [] for key in labels_set}  # Create an empty list for each label
    for node, label in node_to_label.items():
        label_to_node[label].append(node)  # Append node to the corresponding label
    return label_to_node


def create_real_graph(file_path):
    """
    Creates a graph from a file containing edges. Each line in the file represents an edge between two nodes.

    :param file_path: Path to the file containing the edge list
    :return: A NetworkX graph object containing the edges from the file
    """
    graph = nx.Graph()  # Create an empty graph
    delimiter_pattern = re.compile(r'[,\s\t]+')  # Regular expression for splitting by spaces or commas
    with open(file_path, 'r') as data_file:
        for i, line in enumerate(data_file):
            # Split each line by space or comma
            nodes = delimiter_pattern.split(line.strip())
            node1 = int(nodes[0])  # Convert the first node to an integer
            node2 = int(nodes[1])  # Convert the second node to an integer
            if node1 != node2:
                graph.add_edge(node1, node2)  # Add the edge between the two nodes
    return graph


def labeled_graph(graph, labels_file):
    """
    Adds labels to the graph nodes based on a labels file.

    :param graph: The graph to be labeled
    :param labels_file: The file containing node labels (node, label) pairs
    :return: A tuple containing:
        - The updated graph with labeled nodes
        - A dictionary mapping nodes to their labels
    """
    node_to_label = {}
    delimiter_pattern = re.compile(r'[,\s\t]+')  # Regular expression for splitting by spaces or commas
    with open(labels_file, 'r') as data_file:
        for i, line in enumerate(data_file):
            # Split each line by space or comma
            nodes = delimiter_pattern.split(line.strip())
            node = int(nodes[0])  # Extract the node
            label = int(nodes[1])  # Extract the label
            node_to_label[node] = label  # Map the node to its label
            if not graph.has_node(node):
                graph.add_node(node)  # Ensure the node exists in the graph
    return graph, node_to_label


def color_nodes(graph, nodes_per_color=None):
    """
    Assigns a random color (label) to each node in the graph.

    :param graph: The graph whose nodes will be colored
    :param nodes_per_color: The number of nodes to assign to each color (label)
    :return: A dictionary mapping nodes to their assigned color (label)
    """
    if nodes_per_color is None:
        average_degree = 2 * graph.number_of_edges() / graph.number_of_nodes()
        num_colors = int(average_degree)
    else:
        num_colors = int(graph.number_of_nodes() / nodes_per_color)
    node_to_label = {}
    for node in graph.nodes():
        color_node = random.randint(0, num_colors - 1)  # Assign a random color (label)
        node_to_label[node] = color_node
    return node_to_label


def generate_gnp(num_of_colors, p=0.5, nodes_per_color=1000):
    """
    Generates a GNP (Erdős–Rényi) graph with labeled nodes and plants a clique.
    The generated graph is saved as a .gpickle file.

    :param num_of_colors: The number of colors (labels) for the nodes
    :param p: The probability of edge creation between any two nodes (default 0.5)
    :param nodes_per_color: The number of nodes per color (label) (default 1000)
    :return: The generated graph with a planted clique
    """
    label_to_node = {}  # A dictionary mapping labels to nodes that belong to that label.
    node_to_label = {}  # A dictionary mapping nodes to their assigned labels.
    graph = nx.erdos_renyi_graph(nodes_per_color * num_of_colors, p)
    full_colors = []
    for node in graph.nodes():
        color_node = random.randint(0, num_of_colors - 1)
        while color_node in full_colors:
            color_node = random.randint(0, num_of_colors - 1)
        node_to_label[node] = color_node
        if color_node in label_to_node:
            label_to_node[color_node].append(node)
        else:
            label_to_node[color_node] = [node]
        if len(label_to_node[color_node]) == nodes_per_color:
            full_colors.append(color_node)

    graph_with_clique = plant_clique(graph, label_to_node)
    # Save the graph and dictionaries to a .gpickle file
    file_name = "erdos_renyi_graph_" + "p={}".format(p) + "_{}_classes".format(num_of_colors)
    with open(file_name, 'wb') as f:
        pickle.dump((graph_with_clique, node_to_label, label_to_node), f)
    return graph_with_clique


def plant_clique(graph, label_to_node):
    """
    Plants a clique in the graph by selecting one node from each label group and connecting them.

    :param graph: The graph where the clique will be added
    :param label_to_node: A dictionary mapping each label to its nodes
    :return: The graph with the added clique
    """
    clique = []
    for label in label_to_node.keys():
        added_node = label_to_node[label][random.randint(0, len(label_to_node[label]) - 1)]
        clique.append(added_node)
    edge_list = list(combinations(clique, 2))  # Generate all combinations of the clique nodes
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1])  # Add edges between the clique nodes
    return graph


def get_graph_with_properties(graph_type, edges, colors):
    if graph_type == "gnp":
        (k, p) = edges
        nodes_per_color = colors
        # If the graph type is GNP (Erdős–Rényi graph), generate the graph
        generate_gnp(k, p, nodes_per_color)
        # Construct the file name based on the provided parameters
        file_name = f"erdos_renyi_graph_p={p}_{k}_classes"
        # Load the generated graph and dictionaries from the .gpickle file
        with open(file_name, 'rb') as f:
            graph, node_to_label, label_to_node = pickle.load(f)
        # Store the loaded graph in a variable
        updated_graph = graph
    else:
        edges_file = edges
        # Load the real graph using the specified edges file
        graph = create_real_graph(edges_file)
        if graph_type == "real":
            nodes_per_color = colors
            # If no labels file is provided, color the nodes based on the graph's size
            node_to_label = color_nodes(graph, nodes_per_color)
        elif graph_type == "real_colored":
            labels_file = colors
            graph, node_to_label = labeled_graph(graph, labels_file)
        else:
            print('Error: Graph type should be: "real", "real_colored" or "gnp".')
            return
        # Create the label-to-node dictionary based on node-to-label
        label_to_node = create_label_dict(node_to_label)
        # Plant a clique in the real graph
        graph_with_clique = plant_clique(graph, label_to_node)
        # Update the graph variable to include the planted clique
        updated_graph = graph_with_clique
    return updated_graph, node_to_label, label_to_node


if __name__ == '__main__':
    # generate_gnp(11, 0.3)
    graph, node_to_label, label_to_node = get_graph_with_properties("gnp", (4, 0.3), 5)
    print(graph)
