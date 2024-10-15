import numpy as np
from Problems.synthetic_cosh import synthetic
from analysis.analysis import error
from graph.graph import Random
from Optimizers import DOPTIMIZER as dopt
from utilities import utilities as ut
from utilities.plot_utils import plot_exp4
import os


### creating times
seed       = np.random.randint(12345)  
seed       = 8075
np.random.seed(seed)

#### create asynchronous setup 
num_nodes  = 10
dim        = 5
comp_time_dist = 'random_uniform' 

mincomp = np.array([1,1,1,1,1,1,1,1,1,1])
maxcomp = np.array([5,10,15,20,25,30,35,40,45,50])

T_active_exp4, Tv_nodes_exp4, node_comp_time_exp4 = \
    ut.create_computation_time(num_nodes, max_iter=int(1e5), comp_time_dist=comp_time_dist, mean_comp=None,\
                                      min_comp=mincomp, max_comp=maxcomp, variance_comp=None, make_integer=True) 
             
                                            
#### some parameters of the algorithms
learning_rate    = 0.01
depoch    = 12000
rho       = 0.01         
alpha     = 0.1
gamma     = 0.5
eta       = 1.0
expScale  = 1/10.

drop_prob_0  = 0.0
drop_prob_1  = 0.25
drop_prob_2  = 0.5
drop_prob_3  = 0.75


#### Problem setup: parameters of the synthetic functions and constraints. 
prd = synthetic(seed, num_nodes, dim)
error_prd = error(prd,np.zeros(num_nodes),0)


#### Create gossip matrices
zero_row_sum,zero_col_sum,row_stochastic,col_stochastic, N_out, neighbors  = Random(num_nodes, prob=0.8, Laplacian_dividing_factor= 2).directed()


#### Run the optimization algorithms and compute the performance metrics 
np.random.seed(seed)
x_asy_dagp1, _, _, _, _, Delay_mat_dagp_1 = \
    dopt.Asy_DAGP(T_active_exp4, Tv_nodes_exp4, prd, zero_row_sum, zero_col_sum, learning_rate, depoch, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=True, drop_prob=drop_prob_1)
np.random.seed(seed)
x_asy_dagp2, _, _, _, _, Delay_mat_dagp_2 = \
    dopt.Asy_DAGP(T_active_exp4, Tv_nodes_exp4, prd, zero_row_sum, zero_col_sum, learning_rate, depoch, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=True, drop_prob=drop_prob_2)
np.random.seed(seed)
x_asy_dagp3, _, _, _, _, Delay_mat_dagp_3 = \
    dopt.Asy_DAGP(T_active_exp4, Tv_nodes_exp4, prd, zero_row_sum, zero_col_sum, learning_rate, depoch, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=True, drop_prob=drop_prob_3)
np.random.seed(seed)
x_asy_dagp0, _, _, _, _, Delay_mat_dagp_0 = \
    dopt.Asy_DAGP(T_active_exp4, Tv_nodes_exp4, prd, zero_row_sum, zero_col_sum, learning_rate, depoch, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=True, drop_prob=drop_prob_0)


f_asy_dagp0 = error_prd.cost_path(np.sum(x_asy_dagp0,  axis=1)/num_nodes) 
f_asy_dagp1 = error_prd.cost_path(np.sum(x_asy_dagp1,  axis=1)/num_nodes) 
f_asy_dagp2 = error_prd.cost_path(np.sum(x_asy_dagp2,  axis=1)/num_nodes) 
f_asy_dagp3 = error_prd.cost_path(np.sum(x_asy_dagp3,  axis=1)/num_nodes) 


#### save data and plot results
plot_exp4(T_active_exp4, f_asy_dagp0, f_asy_dagp1, f_asy_dagp2, f_asy_dagp3, current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder='exp4', plot_iter=depoch)


