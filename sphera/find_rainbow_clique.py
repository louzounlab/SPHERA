import argparse
import pickle
import numpy as np
import time
import random
import process_graph


def bron_kerbosch(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, found_cliques=[],
                  start_time=None, timeout=600):
    """
    The Bron-Kerbosch algorithm for finding maximal cliques in a graph.

    :param graph: The input graph.
    :param labels: List of all labels in the graph.
    :param label_to_node_: Dictionary mapping labels to nodes.
    :param potential_clique: Nodes currently being considered for the clique.
    :param remaining_nodes: Nodes still available for inclusion in the clique.
    :param skip_nodes: Nodes already processed.
    :param found_cliques: List of cliques found so far.
    :param start_time: The time the function started (used for timeout).
    :param timeout: Maximum allowed time in seconds before stopping (default is 600 seconds).
    :return: List of all cliques found in the graph up to the timeout.
    """
    if start_time is None:
        start_time = time.time()
    # Check for timeout, stop and return found cliques if timeout reached
    if time.time() - start_time > timeout:
        return found_cliques
    # Base case: no more remaining nodes and no nodes to skip, so add the clique to the list
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        found_cliques.append(potential_clique)
        return found_cliques
    # If we already have a clique with all labels, return the result
    if len(potential_clique) == len(label_to_node_):
        found_cliques.append(potential_clique)
        return found_cliques

    for node in remaining_nodes:
        if time.time() - start_time > timeout:
            return found_cliques
        # Extend the current potential clique by adding the current node
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        # Recalculate the remaining nodes and skip list based on neighbors of the current node
        new_skip_list = [n for n in skip_nodes if n in list(graph.neighbors(node))]
        found_cliques = bron_kerbosch(
            graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
            new_skip_list, found_cliques, start_time, timeout
        )
        # If we found a maximal clique (with all labels), return early
        if len(found_cliques[-1]) == len(label_to_node_):
            return found_cliques

        # Remove processed node
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return found_cliques


def fixed_order_barrier(graph, node_to_label, label_to_node, labels_list, potential_clique, remaining_nodes,
                        added_labels, nodes_in_label_degree, name, max_cliques_founded=[], gate_change=True, start_time=None,
                        time_limit=600, greedy=False, gate=False, greedy_size=0, first=True):
    """
    Fixed-order algorithm for finding max rainbow clique.

    :param graph: The graph to search for cliques.
    :param node_to_label: Dictionary mapping nodes to their labels.
    :param label_to_node: Dictionary mapping labels to nodes.
    :param labels_list: List of all labels in the graph, ordered by priority.
    :param potential_clique: The current clique being built.
    :param remaining_nodes: Nodes that can still be added to the clique.
    :param added_labels: Labels that have already been included in the current clique.
    :param nodes_in_label_degree: Dictionary mapping labels to nodes sorted by their degree.
    :param max_cliques_founded: List of cliques found so far.
    :param gate_change: Boolean that changes the 'gate' condition.
    :param start_time: The starting time to monitor for time limits.
    :param time_limit: Maximum time limit for the search (in seconds).
    :param gate: Boolean flag for controlling the search by neighbors of cliques.
    :param greedy_size: Size of the greedy clique.
    :param first: Boolean flag for the first run.
    :return: The largest clique found, the size of the greedy clique, and updated control flags.

     """
    if name is None:
        filename = f"all_way_clique_{len(labels_list)}.txt"
    else:
        filename = f"all_way_clique_{name}.txt"
    # Write to a file all the partial cliques found for analyzing.
    with open(filename, 'a') as f:
        f.write(f"{potential_clique}\n")
    # Check if the time limit has been exceeded
    if time.time() - start_time > time_limit:
        print("Time limit exceeded")
        return max_cliques_founded, greedy_size, first, gate
    # Search by greedy method and greedy method got stopped
    if greedy and greedy_size > 0:
        return max_cliques_founded, greedy_size, first, gate
    # Check if the current clique is complete or no more nodes are available
    if len(potential_clique) == len(labels_list) or len(remaining_nodes) == 0:
        return potential_clique, greedy_size, first, gate
    # check if found better clique than could be found
    # future_possible_labels = [node_to_label[n] for n in remaining_nodes]
    # if len(set(future_possible_labels)) + len(added_labels) <= len(max_cliques_founded):
    #     return max_cliques_founded, greedy_size, first, gate
    # Determine the next label to work on (i.e., label not yet added to the clique)
    next_label = -1
    for label in labels_list:
        if label not in added_labels:
            next_label = label
            break
    if next_label == -1:
        return max_cliques_founded, greedy_size, first, gate
    else:
        # Get nodes associated with the next label, sorted by degree
        nodes = nodes_in_label_degree[next_label]
        potential_nodes_in_label = [node for node in nodes if node in remaining_nodes]
    nodes_to_try = potential_nodes_in_label.copy()
    # If no nodes left to try, update the 'gate' and 'greedy_size' if necessary
    if len(nodes_to_try) == 0:
        if first is True:
            # Greedy approach failed, save the greedy clique size.
            greedy_size = len(potential_clique)
            first = False
            # Search by greedy method
            if greedy:
                max_cliques_founded = potential_clique
                return max_cliques_founded, greedy_size, first, gate
        if gate_change is True:
            # From the point greedy failed, add the check of the gate.
            gate = True
    for _ in range(len(potential_nodes_in_label)):
        # Check if the time limit has been exceeded
        if time.time() - start_time > time_limit:
            print("Time limit exceeded in for")
            return max_cliques_founded, greedy_size, first, gate
        # If no nodes left to try, update the 'gate' and 'greedy_size' if necessary
        if len(nodes_to_try) == 0:
            if first is True:
                # Greedy approach failed, save the greedy clique size.
                greedy_size = len(potential_clique)
                first = False
                # Search by greedy method
                if greedy:
                    max_cliques_founded = potential_clique
                    return max_cliques_founded, greedy_size, first, gate
            if gate_change is True:
                # From the point greedy failed, add the check of the gate.
                gate = True
        node = nodes_to_try[0]
        nodes_to_try.remove(node)
        # Extend the current clique with the selected node and filter the remaining nodes
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(next_label)
        if gate:
            # Check if the current clique can expand to the max clique by checking the number of labels of the clique
            # and its neighbors.
            colors = [node_to_label[x] for x in new_remaining_nodes] + new_added_labels
            if len(set(colors)) < len(labels_list):
                cliques_founded = potential_clique
            else:
                # Recursive call to extend the clique further
                cliques_founded, greedy_size, first, gate = fixed_order_barrier(graph, node_to_label, label_to_node,
                                                                                labels_list, new_potential_clique,
                                                                                new_remaining_nodes, new_added_labels,
                                                                                nodes_in_label_degree, name,
                                                                                max_cliques_founded, gate_change,
                                                                                start_time, time_limit, greedy, gate,
                                                                                greedy_size, first)
        else:
            # Proceed without gate conditions
            cliques_founded, greedy_size, first, gate = fixed_order_barrier(graph, node_to_label, label_to_node,
                                                                            labels_list, new_potential_clique,
                                                                            new_remaining_nodes, new_added_labels,
                                                                            nodes_in_label_degree, name,
                                                                            max_cliques_founded,
                                                                            gate_change, start_time, time_limit, greedy,
                                                                            gate, greedy_size, first)
        # If the clique found has all the labels, return it
        if len(cliques_founded) == len(labels_list):
            return cliques_founded, greedy_size, first, gate
        elif len(cliques_founded) > len(max_cliques_founded):
            max_cliques_founded = cliques_founded
        remaining_nodes.remove(node)  # Remove node from remaining pool
    return max_cliques_founded, greedy_size, first, gate


def degree_nodes_in_label(graph, label_to_node, label):
    """
    Returns the nodes associated with a specific label, sorted in descending order of their degree.

    :param graph: The given graph
    :param label_to_node: Dictionary mapping each label to a list of nodes
    :param label: The label to get nodes for
    :return: List of nodes sorted by degree in descending order or -1 if no nodes are found
    """
    # Get all nodes associated with the label
    nodes_in_label = [n for n in label_to_node[label]]
    # Get the degree of each node in the label
    nodes_rank = [val for (node, val) in graph.degree(nodes_in_label)]
    nodes_to_order = nodes_in_label.copy()
    if len(nodes_to_order) == 0:
        return -1  # Return -1 if no nodes are available for the label
    # Get the indices that would sort the nodes by their degree in descending order
    sorted_indices = np.argsort(nodes_rank)[::-1]
    # Order the nodes based on their ranks (degrees)
    ordered_nodes = [nodes_to_order[i] for i in sorted_indices]
    return ordered_nodes


def rc_detection(graph, node_to_label, label_to_node, heuristic=True, gate_change=True, time_limit=600, greedy=False, name=None):
    """
    Detection of max rainbow clique using the SPHERA algorithm.

    :param graph: Input graph
    :param node_to_label: Dictionary mapping nodes to labels
    :param label_to_node: Dictionary mapping labels to their respective nodes
    :param heuristic: Whether to use a heuristic (based on degree) to order the labels and nodes.
    :param gate_change: Boolean to toggle the use of the 'gate' condition during clique construction
    :param time_limit: Time limit (in seconds) for the function execution
    :return: The largest clique found and the greedy clique size
    """
    # Start the timer for time-limited execution
    start_time = time.time()
    # Get all labels in the graph
    all_labels = list(label_to_node.keys())
    to_remove = {}
    num_labels = len(all_labels)
    if gate_change:
        for v in list(graph.nodes):
            if v in graph.nodes:
                neighbor_labels = {node_to_label[neighbor] for neighbor in graph.neighbors(v) if neighbor in node_to_label}
                if len(neighbor_labels) < num_labels - 1:
                    label_v = node_to_label[v]
                    if label_v not in to_remove:
                        to_remove[label_v] = []
                    to_remove[label_v].append(v)
                    node_to_label.pop(v, None)
                    graph.remove_node(v)
        for label in to_remove:
            set_label = set(to_remove[label])
            current = label_to_node[label]
            update = [node for node in current if node not in set_label]
            label_to_node[label] = update
        if len(node_to_label) == num_labels and is_clique(graph, graph.nodes, label_to_node, node_to_label):
            clique = list(label_to_node.values())
            return clique, -1
    if heuristic:
        # Calculate the average degree of nodes for each label
        average_rank = {label: np.mean(list(graph.degree(label_to_node[label])))
                        for label in all_labels}
        # Sort the labels based on the average degree (ascending order)
        labels_degree = sorted(average_rank, key=average_rank.get)
        # Initialize a dictionary to store nodes for each label sorted by degree
        nodes_in_label_degree = {label: [] for label in all_labels}
        # Populate the dictionary with nodes sorted by their degree within each label
        for label in all_labels:
            nodes_in_label_degree[label] = degree_nodes_in_label(graph, label_to_node, label)
    else:
        # If no heuristic, randomly shuffle the labels
        labels_degree = all_labels.copy()
        random.shuffle(labels_degree)
        # Initialize the dictionary for nodes in each label
        nodes_in_label_degree = {label: [] for label in all_labels}
        # Populate the dictionary with randomly shuffled nodes for each label
        for label in all_labels:
            potential_nodes_in_label = [n for n in label_to_node[label]]
            random.shuffle(potential_nodes_in_label)
            nodes_in_label_degree[label] = potential_nodes_in_label
    # Call the 'fixed_order_barrier' function to construct the clique
    cliques_founded, greedy_size, _, gate = fixed_order_barrier(
        graph, node_to_label, label_to_node, labels_degree, [], list(graph.nodes()), [],
        nodes_in_label_degree, name, [], gate_change, start_time, time_limit, greedy, name
    )
    if greedy_size == 0:
        # If the greedy method detected the entire clique, set greedy_size to the size of clique.
        greedy_size = len(cliques_founded)
    return cliques_founded, greedy_size


def is_clique(g, nodes, label_to_node_, node_to_label_):
    """

    Checks if a set of nodes forms a complete subgraph (clique) and, optionally, verifies that all
    labels are different in the case of a triplet.
    :param g: The input graph.
    :param nodes: List of nodes to check if they form a clique.
    :param label_to_node_: A dictionary mapping labels to nodes.
    :param node_to_label_: A dictionary mapping nodes to labels.
    :return: True if the nodes form a clique with the appropriate label condition, False otherwise.
    """
    # Create a subgraph from the given nodes
    h = g.subgraph(nodes)
    n = len(nodes)
    # Check if the subgraph is a complete graph (i.e., every node is connected to every other node)
    if int(h.number_of_edges()) == int(n * (n - 1) / 2):  # Formula for complete graph edges
        if len(nodes) == len(label_to_node_):
            my_labels = []
            # Collect the labels of the nodes
            for node in nodes:
                my_labels.append(node_to_label_[node])
            # Check that all labels are unique
            if len(set(my_labels)) == len(label_to_node_):  # All labels should be different
                return True
            else:
                return False
        else:
            # If the number of nodes does not match the number of labels, return False
            return False
    else:
        # If the subgraph is not complete, return False
        return False


def parse_args():
    """
    Parses command-line arguments for the graph generation and analysis program.

    This function allows the user to specify the type of graph ('gnp' or 'real'),
    parameters for graph generation (such as the number of classes, edge probability,
    and nodes per color), and options for heuristic search and gate usage.

    For GNP graphs, 'k' (number of classes), 'p' (edge probability), and 'nodes_per_color'
    (number of nodes per color) are used, while for real_graphs, edge and label files must be
    specified.

    :return: args: Parsed arguments containing graph type, parameters, and flags for options.
    """
    parser = argparse.ArgumentParser(description="Process graph type and options")
    # Flag 1: Choose between GNP or real graph
    parser.add_argument('--graph_type', choices=['gnp', 'real'], required=True,
                        help="Specify the graph type: 'gnp' for GNP graph, 'real' for a real graph")
    # Flag 2: If GNP, get tuple (k, p, nodes_per_color)
    parser.add_argument('--k', type=int, help="Number of classes (required for GNP)", default=8)
    parser.add_argument('--p', type=float, help="Probability for edge creation in GNP graph", default=0.3)
    parser.add_argument('--nodes_per_color', type=int, help="Number of nodes per color in GNP graph", default=100)
    # If 'real' graph type, specify file directory for edge list
    parser.add_argument('--edges_file', type=str, help="File of edges in real graph (required for 'real')")
    # If 'real' graph type, specify the file for node labels (optional)
    parser.add_argument('--labels_file', type=str, help="file with node labels (optional,"
                                                        "used with 'real' graph type).")
    def str_to_bool(value):
        """
        Converts a string value to a boolean (True or False).
        Accepts 'True'/'true' or 'False'/'false'.
        """
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected ('True' or 'False').")
    # Flag 3: Heuristic - True/False
    parser.add_argument('--heuristic', type=str_to_bool, default=True, help="Use heuristic search (default: True)")
    # Flag 4: Gate - True/False
    parser.add_argument('--gate', type=str_to_bool, default=True, help="Use gate (default: True)")

    # Parse the arguments from the command line
    args = parser.parse_args()
    # Validate the arguments
    if args.k <= 0:
        raise ValueError("Error: 'k' (number of classes) must be a positive integer.")
    if args.nodes_per_color <= 0:
        raise ValueError("Error: 'nodes_per_color' must be a positive integer.")
    if not (0 <= args.p <= 1):
        raise ValueError("Error: 'p' (edge probability) must be a float between 0 and 1.")
    return args


def main():
    """
    Main function that handles the graph generation and clique detection process based on user-provided command-line
    arguments.

    The function performs the following steps:
    1. Parse Arguments: It uses `parse_args()` to process the command-line input to determine the type of graph
       ('gnp' or 'real') and other parameters such as edge probability, number of classes, nodes per color, etc.

    2. Graph Generation: Based on the user's choice of graph type:
       - GNP (Erdős–Rényi Graph): If 'gnp' is selected, it generates the graph using the specified parameters
         (`k`, `p`, `nodes_per_color`), then loads the graph, node-to-label, and label-to-node mappings from a saved
         `.gpickle` file.
       - Real Graph: If 'real' is selected, it requires the user to specify an edges file. Optionally, it can also take
       a labels file
         for node labeling. It constructs the graph using real-world data, assigns labels to the nodes (either by
         coloring or via a label file), and optionally plants a clique in the graph.

    3. Clique Detection: After loading or generating the graph, it calls the `rc_detection()` function to find the maximum
    rainbow clique, utilizing optional heuristic and gate parameters as specified by the user.

    4. Output: The function prints the maximum rainbow clique found in the graph.
    """
    # Parse the command-line arguments
    args = parse_args()
    # Load the appropriate graph based on the user's selection
    if args.graph_type == 'gnp':
        # If the graph type is GNP (Erdős–Rényi graph), generate the graph
        process_graph.generate_gnp(args.k, args.p, args.nodes_per_color)
        # Construct the file name based on the provided parameters
        file_name = f"erdos_renyi_graph_p={args.p}_{args.k}_classes"
        # Load the generated graph and dictionaries from the .gpickle file
        with open(file_name, 'rb') as f:
            graph, node_to_label, label_to_node = pickle.load(f)
        # Store the loaded graph in a variable
        updated_graph = graph

    elif args.graph_type == 'real':
        # If the graph type is 'real', check if the edges file is provided
        if not args.edges_file:
            print("Error: You must specify a file directory for the real graph.")
            return
        # Load the real graph using the specified edges file
        graph = process_graph.create_real_graph(args.edges_file)
        # If no labels file is provided, color the nodes based on the graph's size
        if not args.labels_file:
            node_to_label = process_graph.color_nodes(graph, args.nodes_per_color)
        else:
            # If a labels file is provided, use it to assign labels to the nodes
            graph, node_to_label = process_graph.labeled_graph(graph, args.labels_file)
        # Create the label-to-node dictionary based on node-to-label
        label_to_node = process_graph.create_label_dict(node_to_label)
        # Plant a clique in the real graph
        graph_with_clique = process_graph.plant_clique(graph, label_to_node)
        # Update the graph variable to include the planted clique
        updated_graph = graph_with_clique
    else:
        # If no valid graph type is specified, print an error and exit
        print("Error: You must specify the type of graph.")
        return
    # Call the sphera function to find the maximum rainbow clique with the specified parameters
    max_rainbow_clique, _ = rc_detection(updated_graph, node_to_label, label_to_node, args.heuristic, args.gate)
    # Output the found clique
    print("Clique found:", max_rainbow_clique)


if __name__ == '__main__':
    main()
