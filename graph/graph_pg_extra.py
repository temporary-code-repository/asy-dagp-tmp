
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def generate_graph_and_matrices(n, p, plot=False):
    """
    Generates a random graph and computes the Laplacian, Incidence, and Diagonal matrices.
    
    :param n: Number of nodes
    :param p: Probability of edges
    :return: Laplacian matrix L, Incidence matrix C, Diagonal matrix D, Weight matrix W
    """
    # Create a random undirected graph
    G = nx.erdos_renyi_graph(n, p)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        raise ValueError("The generated graph is not connected. Please try again with a different probability or number of nodes.")
    
    # Assign weights to the edges
    for u, v in G.edges():
        G[u][v]['weight'] = 1  # Example: Assigning weight 1 to all edges
    
    # Calculate the Laplacian matrix L
    L = nx.laplacian_matrix(G).toarray()
    
    # Calculate the Incidence matrix C
    C = -nx.incidence_matrix(G, oriented=True).T.toarray()
    
    # Calculate the Weight matrix W as Identity matrix - Laplacian / (2 * max_degree)
    max_degree = max(dict(G.degree(weight='weight')).values())
    W = np.eye(n) - L / (2 * max_degree)
    D = np.diag([np.sqrt(W[i, j]/2) for i, j in G.edges])
    V = np.matmul(D,C)

    # Get the neighbors of each node including the node itself
    neighbors = {node: set([node]) | set(G.neighbors(node)) for node in G.nodes}

    # Get the list of edges connected to each node
    edges_connected = {node: set(G.edges(node)) for node in G.nodes}


    # Print the number of created edges from the total possible number of edges
    created_edges = G.number_of_edges()
    # total_possible_edges = n * (n - 1) // 2  # For undirected graph
    # print(f"Created {created_edges} edges from {total_possible_edges} possible edges.")
    

  # Initialize a dictionary to store the indices of edges connected to each node
    edge_indices = {node: [] for node in G.nodes}
    
    # Assign indices to the edges and store them in the dictionary
    for index, (u, v) in enumerate(G.edges()):
        if v > u:  # Ensure that j > i
            edge_indices[u].append(index)
        elif u > v:  # Ensure that j > i
            edge_indices[v].append(index)
    

    n = len(G.nodes)
    H = np.zeros((n, n))  # Initialize Metropolis matrix
    
    for i in G.nodes:
        for j in G.neighbors(i):
            H[i, j] = 1 / (1 + max(G.degree[i], G.degree[j]))
        
        # Set diagonal elements
        H[i, i] = 1 - np.sum(H[i, :])
    
    neighbors = 1*(W>0)
    zr = np.eye(n)-W
    zc = np.eye(n)-W

    # Plot the graph
    if plot:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='black', linewidths=1, font_size=15, connectionstyle='arc3,rad=0.1')
        
        # Label the edges with their indices
        edge_labels = {(u, v): idx for idx, (u, v) in enumerate(G.edges())}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.show()
    
    return L, C, W, D, V, neighbors, edges_connected, edge_indices, created_edges, H, neighbors, zr, zc




####  Example usage:
# n = 5  # Number of nodes
# p = 0.8  # Probability of edges
# L, C, W, D, V, neighbors, edges_connected, edge_indices, num_edges, H = generate_graph_and_matrices(n, p, plot=False)

# print("Laplacian Matrix L:\n", L)
# print("Incidence Matrix C:\n", C)
# print("Diagonal Matrix D:\n", D)
# print("Weight Matrix W:\n", W)
# print("Metropolis Matrix H:\n", H)
# print("Neighbors of each node including the node itself:\n", neighbors)
# print("List of edges connected to each node:\n", edges_connected)
# print("Indices of edges connected to each node (j > i):\n", edge_indices)