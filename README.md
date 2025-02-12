<p align="center">
    <img src="https://github.com/user-attachments/assets/2048ed5e-01aa-4db2-bbcb-7613eb1624ff" alt="SPHERA">
</p>

<p align="center">
    <a href="https://img.shields.io/badge/python-100%25-blue">
        <img alt="python" src="https://img.shields.io/badge/python-100%25-blue">
    </a>
    <a href="https://img.shields.io/badge/license-MIT-blue">
        <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
    </a>

**SPHERA**- Search Space Limitation Efficient Rainbow Clique Algorithm algorithm to find rainbow cliques using a
combination of greedy growth, backtracking, and an efficient minimization of the search space. 

Our python package contains the main function **sphera** and additional methods and plots.

## Table of Contents

-  [Installation](#installation)
-  [Quick tour](#quick_tour)

[//]: # (-  [Examples]&#40;#examples&#41;)

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sphera.
```bash
pip install sphera
```

## Quick tour
To immediately use our package, you only need to run a single function.<br>
For analysis on real-world data, first you have to prepare a csv or txt file which contains the edges of the required graph.
### Real-world data (non-colored graph)
The file containing the list of edges should look like this:
```csv
FirstID, SecondID
0	1
1	0
1	2
1	3
1	4
1	5
1	6
1	7
1	8
1	9
1	10
1	11
```
- FirstID: ID of first vertex in the edge.
- SecondID: ID of second vertex in the edge.

A few notes:
- The file is required to have no header.
- Between vertices could be tab or comma.
- The input type in each tab should be an integer.
- 
After preparing the files you can run SPHERA throw the command line:
```
python .\find_rainbow_clique.py --type real --edges_file file1 --nodes_per_color 1000 --heuristic True --gate True 

```
Where:
- `type`: real (for the option of real graphs).
- `edges_file`: A path to a csv or txt file with columns: 1) id of first vertex (integer). 2) id of second vertex.
    Assuming columns are separated with , or tab and no whitespaces in the csv file.
- `nodes_per_color`: (optional, default is the average degree) specify the number of vertices in each color (integer).
- `heuristic`: True if you want SPHERA to use heuristics, False otherwise.
- `gate`: True if you want SPHERA to check the neighbors of clique at each stage, False otherwise.

Returns: the vertices in the maximum rainbow clique.

### Real-world data (colored graph)
For colored graphs, 2 files are required. The first file contains the edges as before, the second contains the labels of the vertices.
The file containing the labels of vertices should look like this:
```csv
1,1
2,2
3,1
4,2
5,2
6,2
7,2
8,2
9,2
10,2
```
- VertexID: ID of the vertex.
- Label: label of the vertex.

A few notes:
- The file is required to have no header.
- Between the vertex and label could be tab or comma.
- The input type in each tab should be an integer.

After preparing the files you can run SPHERA throw the command line:
```
python .\find_rainbow_clique.py --type real --edges_file file1 --labels_file file2 --heuristic True --gate True 

```
Where:
- `type`: real (for the option of real graphs).
- `edges_file`: A path to a csv or txt file with columns: 1) id of first vertex (integer). 2) id of second vertex.
    Assuming columns are separated with , or tab and no whitespaces in the csv file.
- `labels_file`: A path to a csv or txt file with columns: 1) id of vertex (integer). 2) label of the vertex (integer).
    Assuming columns are separated with , or tab and no whitespaces in the csv file.
- `heuristic`: True if you want SPHERA to use heuristics, False otherwise.
- `gate`: True if you want SPHERA to check the neighbors of clique at each stage, False otherwise.

Returns: the vertices in the maximum rainbow clique.

#### UMAT with uncertainty
```
python .\find_rainbow_clique.py --type gnp --k 9 --p 0.3 --nodes_per_color 1000 --heuristic True --gate True 

```
Where:
- `type`: gnp (for the option of G(n,p) graphs).
- `k`: The number of colors in the graph.
- `p`: The probability of existence of an edge in the graph.
- `nodes_per_color`: Specify the number of vertices in each color (integer).
- `heuristic`: True if you want SPHERA to use heuristics, False otherwise.
- `gate`: True if you want SPHERA to check the neighbors of clique at each stage, False otherwise.

Returns: the vertices in the maximum rainbow clique.


You can find the scripts and the simulated data in:
```bash
├───SPHERA
│   ├───plots
│   │   └───greedy_percentage.py
│   │   └───natual_threshold.py
│   │   └───prob_next_vertex.py
│   │   └───real_graph_plots.py
│   └───find_rainbow_clique.py
│   └───process_graph.py

```