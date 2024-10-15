########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################
import os
import numpy as np

def monitor(name, current, total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )


def active_set(t_active, Tv_nodes):
    """
    Identify the nodes that are activated at a specific time.

    Parameters:
    - num_nodes: Total number of nodes.
    - t_active: The specific time at which nodes' activation status is checked.
    - Tv_nodes: A 2D array where each row represents a node and contains the times at which the node gets activated.

    Returns:
    - List of nodes activated at t_active.
    """

    activated_nodes = np.where(Tv_nodes == t_active)[0]

    return activated_nodes.tolist()


### for pg_extra algorithm
def active_set_continuous(num_nodes, t_active, Tv_nodes, threshold=1e-4):
    active_set = []
    for tmp in range(num_nodes):
        # Check if the absolute difference between t_active and any element in Tv_nodes[tmp] is less than the threshold
        if any(abs(t_active - t) < threshold for t in Tv_nodes[tmp]):
            active_set.append(tmp)
    return active_set


def create_computation_time(num_nodes, max_iter, comp_time_dist, mean_comp, min_comp=None, max_comp=None, variance_comp=None, make_integer=True):
    """
    Generate computation times for nodes across iterations and determine activation times.

    Parameters:
    - num_nodes: Number of nodes.
    - max_iter: Maximum number of iterations.
    - comp_time_dist: Distribution type ('normal', 'exp', 'random_uniform') for computation times.
    - mean_comp: Array of mean computation times for each node.
    - min_comp: Array of minimum computation times for each node (for 'random_uniform').
    - max_comp: Array of maximum computation times for each node (for 'random_uniform').
    - variance_comp: Array of variances for computation times (for 'normal' distribution).
    - make_integer: Round computation times to the nearest integer.

    Returns:
    - T_active: Unique times of activation across all nodes.
    - Tv_nodes: Activation times for each node.
    - node_comp_time: Computation duration for each iteration at each node.
    """
    
    # Initialize node computation time matrix
    node_comp_time = np.zeros((num_nodes, max_iter))

    # Generate computation times based on specified distribution
    if comp_time_dist == 'normal':
        node_comp_time = np.random.normal(mean_comp[:, None], variance_comp[:, None], (num_nodes, max_iter))
        node_comp_time[node_comp_time < 1] = 1  # Ensure minimum computation time of 1
    elif comp_time_dist == 'exp':
        node_comp_time = np.random.exponential(mean_comp[:, None], (num_nodes, max_iter))
    elif comp_time_dist == 'random_uniform':
        for i in range(num_nodes):
            node_comp_time[i] = np.random.uniform(min_comp[i], max_comp[i], max_iter)

    # Prepend a zero column and round up if needed
    node_comp_time = np.hstack((np.zeros((num_nodes, 1)), node_comp_time))
    if make_integer:
        node_comp_time = np.ceil(node_comp_time)

    # Calculate activation times for each node
    Tv_nodes = np.cumsum(node_comp_time, axis=1)

    # Determine unique activation times across all nodes
    T_active = np.unique(Tv_nodes)

    return T_active, Tv_nodes, node_comp_time


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)


