import pickle
import networkx as nx
import random
from itertools import combinations



def generate_graph(num_of_colors, p=0.5, nodes_per_color=1000):
    if p is None:
        p = 1/(3*nodes_per_color)
        print(p)
    label_to_node = {}
    node_to_label = {}
    """
    label_to_node_: a dictionary of label and the nodes in this label
    node_to_label_: a dictionary of node and it's label
    """
    G = nx.erdos_renyi_graph(nodes_per_color * num_of_colors, p)
    for node in G.nodes():
        color_node = random.randint(0, num_of_colors - 1)
        node_to_label[node] = color_node
        if color_node in label_to_node:
            label_to_node[color_node].append(node)
        else:
            label_to_node[color_node] = [node]
    # label_to_node = {key: label_to_node[key] for key in range(0, num_of_colors)}
    planted_clique = plant_clique(G, label_to_node, node_to_label)
    temp_for_save = (G, node_to_label, label_to_node)
    # nx.write_gpickle(temp_for_save, "erdos_renyi_graph_p=0.5" + "_{}_classes".format(num_of_colors))
    #nx.write_gpickle(temp_for_save, "erdos_renyi_graph_" + "p={}".format(p) + "_{}_classes".format(num_of_colors))
    # Save the graph and dictionaries to a .gpickle file
    file_name = "erdos_renyi_graph_" + "p={}".format(p) + "_{}_classes".format(num_of_colors)
    with open(file_name, 'wb') as f:
        pickle.dump((G, node_to_label, label_to_node), f)
    return temp_for_save


def new_generate_graph(num_of_colors, p=0.5, nodes_per_color=1000):
    if p is None:
        p = 1/(3*nodes_per_color)
        print(p)
    label_to_node = {}
    node_to_label = {}
    """
    label_to_node_: a dictionary of label and the nodes in this label
    node_to_label_: a dictionary of node and it's label
    """
    G = nx.erdos_renyi_graph(nodes_per_color * num_of_colors, p)
    full_colors = []
    for node in G.nodes():
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

    # label_to_node = {key: label_to_node[key] for key in range(0, num_of_colors)}
    planted_clique = plant_clique(G, label_to_node, node_to_label)
    print(planted_clique)
    temp_for_save = (G, node_to_label, label_to_node)
    # nx.write_gpickle(temp_for_save, "erdos_renyi_graph_p=0.5" + "_{}_classes".format(num_of_colors))
    #nx.write_gpickle(temp_for_save, "erdos_renyi_graph_" + "p={}".format(p) + "_{}_classes".format(num_of_colors))
    # Save the graph and dictionaries to a .gpickle file
    file_name = "erdos_renyi_graph_" + "p={}".format(p) + "_{}_classes".format(num_of_colors)
    with open(file_name, 'wb') as f:
        pickle.dump((G, node_to_label, label_to_node), f)
    return planted_clique

def plant_clique(G, label_to_node, node_to_label):
    clique = []
    for label in label_to_node.keys():
        added_node = label_to_node[label][random.randint(0, len(label_to_node[label]) - 1)]
        clique.append(added_node)
    print(clique)
    edge_list = list(combinations(clique, 2))
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])

    return clique


if __name__ == '__main__':
    #for i in range(10, 51):
    print(nx.__version__)
    generate_graph(4, 0.1)

