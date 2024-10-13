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

def find_next_label(graph, labels_to_node, added_labels, labels_list):
    average_rank = {label: np.mean(list(graph.degree(labels_to_node[label])))
                    for label in labels_list if label not in added_labels}
    return min(average_rank, key=average_rank.get)


def step_back(graph, node_to_label, label_to_node, labels_list, potential_clique, remaining_nodes,
              added_labels, max_cliques_founded=[]):

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
    # check if success
    if len(potential_clique) == len(labels_list) or len(remaining_nodes) == 0:
        return potential_clique
    # check if found better clique than could be found
    future_possible_labels = [node_to_label[n] for n in remaining_nodes]
    if len(set(future_possible_labels)) + len(added_labels) <= len(max_cliques_founded):
        return max_cliques_founded
    # check which label will be added next
    min_label = find_next_label(graph, label_to_node, added_labels, labels_list)
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node[min_label] if n in remaining_nodes]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    nodes_to_try = potential_nodes_in_label.copy()
    for _ in range(len(potential_nodes_in_label)):
        if len(nodes_to_try) == 0:
            break
        # take max node and remove from potential
        max_node_ind = np.argmax(nodes_rank)
        node = nodes_to_try[max_node_ind]
        nodes_rank.remove(nodes_rank[max_node_ind])
        nodes_to_try.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(min_label)
        cliques_founded = step_back(graph, node_to_label, label_to_node, labels_list, new_potential_clique,
                                    new_remaining_nodes, new_added_labels, max_cliques_founded)
        if len(cliques_founded) == len(labels_list):
            return cliques_founded
        elif len(cliques_founded) > len(max_cliques_founded):
            max_cliques_founded = cliques_founded
        remaining_nodes.remove(node)
    return max_cliques_founded
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

def get_next_node(graph, label_to_node, left_nodes, added_labels):
    """
    get colors list in clique.
    return next node to be added.
    if no nodes in next color return -1.
    return node
    """

    # check which label will be added next
    next_color = find_next_label(graph, label_to_node, added_labels, list(label_to_node.keys()))
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node[next_color] if n in left_nodes]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    nodes_to_try = potential_nodes_in_label.copy()
    if len(nodes_to_try) == 0:
        return -1
    max_node_ind = np.argmax(nodes_rank)
    node = nodes_to_try[max_node_ind]
    return node


def initial_clique(graph, node_to_label, label_to_node, mode=1, labels_degree=None, nodes_in_label_degree=None):
    """
    add to clique get next node
    un till get -1.
    check_barrier(current_clique, left_nodes) ????
    """
    added_labels = []
    current_clique = []
    neighbor_nodes = list(graph.nodes())

    if mode == 1:
        added_node = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
    if mode == 2:
        next_label = -1
        for label in labels_degree:
            if label not in added_labels:
                next_label = label
                break
        if next_label == -1:
            added_node = -1
        else:
            added_node = -1
            nodes = nodes_in_label_degree[next_label]
            for node in nodes:
                if node in neighbor_nodes:
                    added_node = node
                    break
    while added_node != -1:
        current_clique.append(added_node)
        neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, added_node)]
        added_labels.append(node_to_label[added_node])
        if len(current_clique) == len(label_to_node):
            return current_clique
        if mode == 1:
            added_node = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
        if mode == 2:
            next_label = -1
            for label in labels_degree:
                if label not in added_labels:
                    next_label = label
                    break
            if next_label == -1:
                added_node = -1
            else:
                added_node = -1
                nodes = nodes_in_label_degree[next_label]
                for node in nodes:
                    if node in neighbor_nodes:
                        added_node = node
                        break
    # while get_next_node(graph, label_to_node, neighbor_nodes, added_labels) != -1:  # change for run time
    #     added_node = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
    #     current_clique.append(added_node)
    #     neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, added_node)]
    #     added_labels.append(node_to_label[added_node])
    #     if len(current_clique) == len(label_to_node):
    #         return current_clique
    return current_clique


def update_dict(node_to_label, node, neighbor, num_each_color, num_colors, num_neighbors):
    if node_to_label[node] in num_each_color[neighbor]:  # color is connected to node already
        num_each_color[neighbor][node_to_label[node]] += 1
    elif node_to_label[neighbor] == node_to_label[node]:  # same color
        pass
    else:  # new node with this color.
        num_colors[neighbor] += 1
        num_each_color[neighbor][node_to_label[node]] = 1
    num_neighbors[neighbor] += 1


def build_dict(node_to_label, neighbors):
    num_neighbors = {node: 0 for node in neighbors}
    num_colors = {node: 0 for node in neighbors}
    num_each_color = {node: {} for node in neighbors}  # for each key - node, the value is a dict: key is color,
    # value is num from this color
    for node in neighbors:
        for neighbor in neighbors:
            update_dict(node_to_label, node, neighbor, num_each_color, num_colors, num_neighbors)
            update_dict(node_to_label, neighbor, node, num_each_color, num_colors, num_neighbors)
    return num_each_color, num_colors, num_neighbors


def delete_nodes(graph, num_each_color, num_neighbors, num_colors, barrier, nodes_to_delete, node_to_label,
                 remaining_nodes):
    for node in nodes_to_delete:
        if node in graph.nodes():
            for neighbor in remaining_nodes:  # update counting of neighbor
                num_neighbors[neighbor] -= 1
                if node_to_label[node] != node_to_label[neighbor]:
                    num_each_color[neighbor][node_to_label[node]] -= 1
                    if num_each_color[neighbor][node_to_label[node]] == 0:
                        num_colors[neighbor] -= 1
                # check if needed to delete neighbor
                if num_neighbors[neighbor] < barrier:
                    nodes_to_delete.append(neighbor)
                elif num_colors[neighbor] < barrier:
                    nodes_to_delete.append(neighbor)
            # first remove all neighbors need to be removed, then remove node.
            graph.remove_node(node)
    return nodes_to_delete


def check_barrier(graph, node_to_label, label_to_node, clique, left_nodes):
    """
    go over nodes in clique and left nodes
    return left nodes
    """
    num_each_color, num_colors, num_neighbors = build_dict(node_to_label, left_nodes + clique)
    updated_graph = graph.subgraph(left_nodes + clique).copy()
    nodes_to_delete = []
    barrier = len(label_to_node) - 1
    for node in left_nodes:
        if num_neighbors[node] < barrier:
            nodes_to_delete.append(node)
        elif num_colors[node] < barrier:
            nodes_to_delete.append(node)
    # print(nodes_to_delete, "in")
    nodes_to_delete = delete_nodes(updated_graph, num_each_color, num_neighbors, num_colors, barrier, nodes_to_delete,
                                   node_to_label, left_nodes)
    left_nodes = [node for node in left_nodes if node not in nodes_to_delete]
    return left_nodes


def barrier_reduction(graph, node_to_label, label_to_node):
    initial = initial_clique(graph, node_to_label, label_to_node)
    if len(initial) == len(label_to_node):
        return initial
    current_clique = initial
    num_colors = len(label_to_node)
    neighbor_nodes = list(graph.nodes())
    added_labels = []
    for in_clique in current_clique:
        neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
        added_labels.append(node_to_label[in_clique])
    node_to_remove = -1

    while len(current_clique) < num_colors:
        if node_to_remove in neighbor_nodes:
            neighbor_nodes.remove(node_to_remove)
        node_to_add = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
        if node_to_add != -1:
            current_clique.append(node_to_add)
            neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, node_to_add)]
            added_labels.append(node_to_label[node_to_add])
        else:  # delete first node
            node_to_remove = current_clique[0]
            current_clique = current_clique[1:]
            added_labels.remove(node_to_label[node_to_remove])
            neighbor_nodes = list(graph.nodes())
            for in_clique in current_clique:
                neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
        left_nodes = check_barrier(graph, node_to_label, label_to_node, current_clique, neighbor_nodes)
        while len(current_clique + left_nodes) < num_colors:
            print(current_clique)
            node_to_remove = current_clique[0]
            current_clique = current_clique[1:]
            print(current_clique)
            added_labels.remove(node_to_label[node_to_remove])
            neighbor_nodes = list(graph.nodes())
            for in_clique in current_clique:
                neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
            left_nodes = check_barrier(graph, node_to_label, label_to_node, current_clique, neighbor_nodes)
    return current_clique


def check_with_clique_size(graph, node_to_label, label_to_node, clique, left_nodes):
    """
    go over nodes in clique and left nodes
    return left nodes
    """
    num_each_color, num_colors, num_neighbors = build_dict(node_to_label, left_nodes + clique)
    updated_graph = graph.subgraph(left_nodes + clique).copy()
    nodes_to_delete = []
    barrier = len(clique)
    for node in left_nodes:
        if num_neighbors[node] < barrier:
            nodes_to_delete.append(node)
        elif num_colors[node] < barrier:
            nodes_to_delete.append(node)
    # print(nodes_to_delete, "in")
    nodes_to_delete = delete_nodes(updated_graph, num_each_color, num_neighbors, num_colors, barrier, nodes_to_delete,
                                   node_to_label, left_nodes)
    left_nodes = [node for node in left_nodes if node not in nodes_to_delete]
    return left_nodes

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


def barrier_clique_size(graph, node_to_label, label_to_node, mode=1, heuristic=True):

    if mode==1:
        initial = initial_clique(graph, node_to_label, label_to_node)
    else:
        all_labels = list(label_to_node.keys())
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
        initial = initial_clique(graph, node_to_label, label_to_node, 2, labels_degree, nodes_in_label_degree)
    if len(initial) == len(label_to_node):
        return initial, len(initial)
    current_clique = initial
    best_clique = current_clique
    num_colors = len(label_to_node)
    neighbor_nodes = list(graph.nodes())
    added_labels = []
    for in_clique in current_clique:
        neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
        added_labels.append(node_to_label[in_clique])
    node_to_remove = -1

    for i in range(100):
        if len(current_clique) == num_colors:
            break
        if node_to_remove in neighbor_nodes:
            neighbor_nodes.remove(node_to_remove)
        if mode==1:
            node_to_add = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
        if mode==2:
            next_label = -1
            for label in labels_degree:
                if label not in added_labels:
                    next_label = label
                    break
            if next_label == -1:
                node_to_add = -1
            else:
                node_to_add = -1
                nodes = nodes_in_label_degree[next_label]
                for node in nodes:
                    if node in neighbor_nodes:
                        node_to_add = node
                        break
        if node_to_add != -1:
            current_clique.append(node_to_add)
            neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, node_to_add)]
            added_labels.append(node_to_label[node_to_add])
            if len(current_clique) > len(best_clique):
                best_clique = current_clique
        else:  # delete first node
            node_to_remove = current_clique[0]
            current_clique = current_clique[1:]
            # node_to_remove = remove_node_by_neighbors(current_clique, neighbor_nodes)
            # current_clique = current_clique[1:]
            added_labels.remove(node_to_label[node_to_remove])
            neighbor_nodes = list(graph.nodes())
            for in_clique in current_clique:
                neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
        if current_clique == []:
            left_nodes = list(graph.nodes())
        else:
            left_nodes = check_with_clique_size(graph, node_to_label, label_to_node, current_clique, neighbor_nodes)
        colors = [node_to_label[x] for x in current_clique+left_nodes]
        while len(set(colors)) < num_colors:

            node_to_remove = current_clique[0]
            current_clique = current_clique[1:]
            added_labels.remove(node_to_label[node_to_remove])
            neighbor_nodes = list(graph.nodes())
            for in_clique in current_clique:
                neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
            if current_clique == []:
                left_nodes = list(graph.nodes())
            else:
                left_nodes = check_with_clique_size(graph, node_to_label, label_to_node, current_clique, neighbor_nodes)
            colors = [node_to_label[x] for x in current_clique + left_nodes]
    print(current_clique)
    return best_clique, len(initial)

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
    # initial = initial_clique(graph, node_to_label, label_to_node, 2, labels_degree, nodes_in_label_degree)
    # if len(initial) == len(label_to_node):
    #     return initial, len(initial)
    #current_clique = initial
    # neighbor_nodes = list(graph.nodes())
    # added_labels = []
    # for in_clique in current_clique:
    #     neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
    #     added_labels.append(node_to_label[in_clique])
    #left_nodes = check_with_clique_size(graph, node_to_label, label_to_node, current_clique, neighbor_nodes)
    # cliques_founded = fixed_order(graph, node_to_label, label_to_node, labels_degree, current_clique,
    #                             left_nodes, added_labels, nodes_in_label_degree)
    cliques_founded, greedy_size, _, gate = fixed_order(graph, node_to_label, label_to_node, labels_degree, [],
                                  list(graph.nodes()), [], nodes_in_label_degree, [],
                                                        gate_change, start_time, time_limit)
    if gate is False:
        greedy_size = len(cliques_founded)
    print("greedy", greedy_size)
    return cliques_founded, greedy_size

def check_barrier_in_grow(graph, node_to_label, label_to_node):
    check_inside = True
    #initial = initial_clique(graph, node_to_label, label_to_node, check_inside)
    initial = []
    if len(initial) == len(label_to_node):
        return initial
    current_clique = initial
    num_colors = len(label_to_node)
    neighbor_nodes = list(graph.nodes())
    added_labels = []
    for in_clique in current_clique:
        neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, in_clique)]
        added_labels.append(node_to_label[in_clique])
    # now you have neighbors and colors
    for i in range(100):
        if len(current_clique) == num_colors:
            break
        node_to_add = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
        new_neighbors = [node for node in neighbor_nodes if graph.has_edge(node, node_to_add)]
        left_nodes = check_with_clique_size(graph, node_to_label, label_to_node, current_clique + [node_to_add],
                                            new_neighbors)
        colors = [node_to_label[x] for x in current_clique + left_nodes + [node_to_add]]
        while len(set(colors)) < num_colors:
            # new node does not pass barrier
            neighbor_nodes.remove(node_to_add)  # so won't add again
            node_to_add = get_next_node(graph, label_to_node, neighbor_nodes, added_labels)
            if node_to_add == -1:
                break
            new_clique = current_clique + [node_to_add]
            new_neighbors = [neighbor for neighbor in neighbor_nodes if graph.has_edge(neighbor, node_to_add)]
            left_nodes = check_with_clique_size(graph, node_to_label, label_to_node, new_clique,
                                                new_neighbors)
            colors = [node_to_label[x] for x in new_clique + left_nodes]
        current_clique.append(node_to_add)
        neighbor_nodes = [node for node in neighbor_nodes if graph.has_edge(node, node_to_add)]
        added_labels.append(node_to_label[node_to_add])

    return current_clique



if __name__ == '__main__':
    # check if to delete first or last.
    create_erdos_renyi_graph.generate_graph(5, 0.5)
    graph, node_to_label, label_to_node = nx.read_gpickle("erdos_renyi_graph_p=0.5" + "_{}_classes".format(8))
    start1 = time.time()
    print("start")
    clique5 = step_back(graph, node_to_label, label_to_node, list(label_to_node.keys()), [],
                        list(graph.nodes), [])
    print(clique5, "step-back")
    print(is_clique(graph, clique5, label_to_node, node_to_label))
    end1 = time.time()
    print(end1 - start1, "step-back")



    """cliq = check.bron_kerbosch_all(graph.copy(), node_to_label, label_to_node, [],
                                              list(graph.nodes()), [])
    print(cliq, "bron-kerbosch")"""

    start3 = time.time()
    cliq = barrier_clique_size(graph, node_to_label, label_to_node)
    print(cliq, "barrier")
    print(is_clique(graph, cliq, label_to_node, node_to_label))
    end3 = time.time()
    print(end3 - start3, "barrier")

    start33 = time.time()
    cliq22 = barrier_clique_size(graph, node_to_label, label_to_node, 2)
    print(cliq22, "calc once")
    print(is_clique(graph, cliq22, label_to_node, node_to_label))
    end33 = time.time()
    print(end33 - start33, "calc once")

    start2 = time.time()
    cliq = bron_kerbosch(graph.copy(), node_to_label, label_to_node, [],
                                              list(graph.nodes()), [])
    print(cliq, "bron-kerbosch")
    print(is_clique(graph, cliq[0], label_to_node, node_to_label))
    end2 = time.time()
    print(end2 - start2, "bron-kerbosch")

    """start2 = time.time()
        cliq = barrier_reduction(graph, node_to_label, label_to_node)
        print(cliq, "barrier")
        print(is_clique(graph, cliq, label_to_node, node_to_label))
        end2 = time.time()
        print(end2 - start2, "barrier")"""
