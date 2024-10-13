import pickle

import networkx as nx
import numpy as np
import time
import random
import create_erdos_renyi_graph

def is_clique(g, nodes, label_to_node_, node_to_label_, trip=False):
    h = g.subgraph(nodes)
    n = len(nodes)
    if int(h.number_of_edges()) == int(n * (n - 1) / 2):  # complete graph
        if len(nodes) == len(label_to_node_):  # all colors are inside
            if not trip:
                return True
            else:  # why different for triplet.
                my_labels = []
                for node in nodes:
                    my_labels.append(node_to_label_[node])
                if len(set(my_labels)) == len(label_to_node_):  # check that in my labels there are 3 different labels.
                    # print(my_labels)
                    return True
                else:
                    return False
        else:
            return False
    else:
        return False

def bron_kerbosch(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, found_cliques=[]):
    """
    The algorithm using bron-kerbosch algorithm in order to find cliques in graph.
    The algorithm stops if it found clique with all labels (max potential clique)
    :param graph: a graph
    :param labels: a list off all labels in the graph
    :param label_to_node_: a dictionary of label and the nodes in this label
    :param potential_clique:the builded clique until now
    :param remaining_nodes:the potential nodes to be in the clique
    :param skip_nodes: the nodes we already checked and found all cliques with them
    :param found_cliques: the cliques we found until now
    :return: all founded cliques in graph
    """
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        found_cliques.append(potential_clique)
        return found_cliques
    if len(potential_clique) == len(label_to_node_):
        found_cliques.append(potential_clique)
        return found_cliques

    for node in remaining_nodes:
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_skip_list = [n for n in skip_nodes if n in list(graph.neighbors(node))]
        found_cliques = bron_kerbosch(graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
                                       new_skip_list, found_cliques)

        if len(found_cliques[-1]) == len(label_to_node_):
            return found_cliques

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return found_cliques


def fixed_order(graph, node_to_label, label_to_node, labels_list, potential_clique, remaining_nodes,
              added_labels,nodes_in_label_degree, max_cliques_founded=[],gate_change=True, start_time=None, time_limit=600,
                gate=False, greedy_size = 0, first = True):

    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param node_to_label: a dictionary of node, and it's label
     :param label_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the built clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param added_labels: the labels that are in current clique
     :param max_cliques_founded:
     :return: a clique that contain all labels
     """
    with open(f"all_way_clique_{len(labels_list)}.txt", 'a') as f:
        f.write(f"{potential_clique}\n")
    # Check if time limit is exceeded
    if time.time() - start_time > time_limit:
        print("Time limit exceeded")
        return max_cliques_founded, greedy_size, first, gate
    # check if success
    if len(potential_clique) == len(labels_list) or len(remaining_nodes) == 0:
        #print("ccc")
        return potential_clique, greedy_size, first, gate
    # check if found better clique than could be found
    # future_possible_labels = [node_to_label[n] for n in remaining_nodes]
    # if len(set(future_possible_labels)) + len(added_labels) <= len(max_cliques_founded):
    #     print("bbb")
    #     return max_cliques_founded, greedy_size, first, gate
    next_label = -1
    for label in labels_list:
        if label not in added_labels:
            next_label = label
            break
    if next_label == -1:
        return max_cliques_founded, greedy_size, first, gate
    else:
        nodes = nodes_in_label_degree[next_label]
        potential_nodes_in_label = [node for node in nodes if node in remaining_nodes]
    nodes_to_try = potential_nodes_in_label.copy()
    if len(nodes_to_try) == 0:
        if gate_change is True:
            gate = True
        if first is True:
            greedy_size = len(potential_clique)
            first = False
    for _ in range(len(potential_nodes_in_label)):
        if time.time() - start_time > time_limit:
            print("Time limit exceeded in for")
            return max_cliques_founded, greedy_size, first, gate
        if len(nodes_to_try) == 0:
            if gate_change is True:
                gate = True
            if first is True:
                greedy_size = len(potential_clique)
                first = False
            break
        node = nodes_to_try[0]
        nodes_to_try.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(next_label)
        if gate is True:
            colors = [node_to_label[x] for x in new_remaining_nodes] + new_added_labels
            if len(set(colors)) < len(labels_list):
                cliques_founded = potential_clique
            else:
                cliques_founded, greedy_size, first,gate = fixed_order(graph, node_to_label, label_to_node, labels_list, new_potential_clique,
                                            new_remaining_nodes, new_added_labels,nodes_in_label_degree,
                                              max_cliques_founded,gate_change,start_time, time_limit,gate,greedy_size, first)
        else:
            cliques_founded, greedy_size, first, gate = fixed_order(graph, node_to_label, label_to_node, labels_list,
                                                              new_potential_clique,
                                                              new_remaining_nodes, new_added_labels, nodes_in_label_degree,
                                                              max_cliques_founded,gate_change,start_time, time_limit,
                                                                    gate, greedy_size, first)
        if len(cliques_founded) == len(labels_list):
            return cliques_founded, greedy_size, first, gate
        elif len(cliques_founded) > len(max_cliques_founded):
            max_cliques_founded = cliques_founded
        remaining_nodes.remove(node)
    return max_cliques_founded, greedy_size, first, gate


def degree_nodes_in_label(graph, label_to_node, label):
    potential_nodes_in_label = [n for n in label_to_node[label]]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    nodes_to_try = potential_nodes_in_label.copy()
    if len(nodes_to_try) == 0:
        return -1
    # Get the indices that would sort nodes_rank in descending order
    sorted_indices = np.argsort(nodes_rank)[::-1]
    # Order the nodes based on their ranks
    ordered_nodes = [nodes_to_try[i] for i in sorted_indices]
    return ordered_nodes


def fixed_order_barrier(graph, node_to_label, label_to_node, heuristic=True, gate_change=True, time_limit=600):
    all_labels = list(label_to_node.keys())
    start_time = time.time()
    if heuristic:
        average_rank = {label: np.mean(list(graph.degree(label_to_node[label])))
                        for label in all_labels}
        # Sort the labels based on the average rank
        labels_degree = sorted(average_rank, key=average_rank.get)
        nodes_in_label_degree = {label: [] for label in all_labels}
        for label in all_labels:
            nodes_in_label_degree[label] = degree_nodes_in_label(graph, label_to_node, label)
    else:
        labels_degree = all_labels.copy()
        random.shuffle(labels_degree)
        nodes_in_label_degree = {label: [] for label in all_labels}
        for label in all_labels:
            potential_nodes_in_label = [n for n in label_to_node[label]]
            random.shuffle(potential_nodes_in_label)
            nodes_in_label_degree[label] = potential_nodes_in_label
    cliques_founded, greedy_size, _, gate = fixed_order(graph, node_to_label, label_to_node, labels_degree, [],
                                  list(graph.nodes()), [], nodes_in_label_degree, [],
                                                        gate_change, start_time, time_limit)
    if gate is False:
        greedy_size = len(cliques_founded)
    print("greedy", greedy_size)
    return cliques_founded, greedy_size




if __name__ == '__main__':
    k = 8
    create_erdos_renyi_graph.new_generate_graph(k, p=0.3, nodes_per_color=100)
    file_name = f"erdos_renyi_graph_p=0.3_{k}_classes"
    # Load the graph and dictionaries from the .gpickle file
    with open(file_name, 'rb') as f:
        graph, node_to_label, label_to_node = pickle.load(f)
    cliq_true, _ = fixed_order_barrier(graph, node_to_label, label_to_node, False, True)


