#### imports and functions
import numpy as np
from Problems.synthetic_cosh import synthetic
from analysis.analysis import error
from Optimizers import DOPTIMIZER as dopt
from utilities import utilities as ut
import os
from graph.graph_pg_extra import generate_graph_and_matrices
from utilities.plot_utils import plot_exp3


### creating times
seed       = np.random.randint(12345) 
seed       = 45
np.random.seed(seed)


#### create asynchronous setup 
num_nodes  = 10
dim        = 5
comp_time_dist = 'random_uniform' 

mincomp = np.array([1,1,1,1,1,1,1,1,1,1])
maxcomp = np.array([5,10,15,20,25,30,35,40,45,50])

T_active_exp3, Tv_nodes_exp3, node_comp_time_exp3 = \
    ut.create_computation_time(num_nodes, max_iter=int(1e5), comp_time_dist=comp_time_dist, mean_comp=None,\
                                min_comp=mincomp, max_comp=maxcomp, variance_comp=None, make_integer=True) 

                                            
#### some parameters of the algorithms
learning_rate = 0.1
depoch        = 500
rho           = 0.01         
alpha         = 0.1
gamma         = 0.8
eta           = 1.0
expScale      = 1/10.
relax_param   = 0.8*np.ones((num_nodes))


#### Problem setup: parameters of the synthetic functions and constraints. 
prd = synthetic(seed, num_nodes, dim)
error_prd = error(prd,np.zeros(num_nodes),0)


#### Create gossip matrices
L, C, W, D, V, neighbors_list, edges_connected, edge_indices, num_edges, H, neighbors, zero_row_sum, zero_col_sum = generate_graph_and_matrices(num_nodes, 0.8, plot=False)


#### find the optimal solution by running the DAGP algorithm
x_dagp, z_dagp, h_dagp, g_dagp  = dopt.DAGP(prd, zero_row_sum, zero_col_sum, learning_rate, 2*depoch, np.random.randn(num_nodes,dim), rho , alpha, cons = True)
f_dagp = error_prd.cost_path(np.sum(x_dagp,  axis=1)/num_nodes) 
f_opt  = f_dagp[-1]


#### Run the optimization algorithms and compute the performance metrics
x_asy_dagp, _, _, _, _, _ = \
    dopt.Asy_DAGP(T_active_exp3, Tv_nodes_exp3,  prd, zero_row_sum, zero_col_sum, learning_rate, depoch, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, drop_msg=False, drop_prob=0.)

x_pgex, _, _ = \
    dopt.Asy_pg_extra(relax_param, T_active_exp3, Tv_nodes_exp3, prd, W, V, edge_indices, learning_rate, depoch, num_nodes, num_edges, dim, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, drop_msg=False, drop_prob=0.)

f_asy_pgex = abs(error_prd.cost_path(np.sum(x_pgex,      axis=1)/num_nodes) - f_opt)
f_asy_dagp = abs(error_prd.cost_path(np.sum(x_asy_dagp,  axis=1)/num_nodes) - f_opt)


#### save data and plot results: the optimality gap is plotted versus iteration. In the papaer, it is plotted versus the communications. 
plot_exp3(T_active_exp3, f_asy_dagp, f_asy_pgex, current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder='exp3', plot_iter=depoch)
