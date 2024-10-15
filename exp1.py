
import numpy as np
from Problems.synthetic_cosh import synthetic
from analysis.analysis import error
from graph.graph import Random
from Optimizers import DOPTIMIZER as dopt
from utilities import utilities as ut
from utilities.plot_utils import plot_exp1
import os

seed       = np.random.randint(12345)  
seed       = 45
np.random.seed(seed)


#### create asynchronous setup 
num_nodes  = 10
dim        = 5
comp_time_dist = 'random_uniform' 

mincomp = np.array([1,1,1,1,1,1,1,1,1,1])
maxcomp = np.array([5,10,15,20,25,30,35,40,45,50])

T_active_exp1, Tv_nodes_exp1, node_comp_time_exp1 = \
    ut.create_computation_time(num_nodes, max_iter=int(1e5), comp_time_dist=comp_time_dist, mean_comp=None,\
                                      min_comp=mincomp, max_comp=maxcomp, variance_comp=None, make_integer=True) 
                                          

#### some parameters of the algorithms
learning_rate    = 0.01
max_iter_syn     = 60000
max_iter_asyn    = 60000

rho       = 0.01         
alpha     = 0.1
gamma     = 0.5
eta       = 1.0
expScale  = 1/10.
theta_0   = np.random.randn(num_nodes,dim)


#### Problem setup: parameters of the synthetic functions and constraints. 
prd = synthetic(seed, num_nodes, dim)
error_prd = error(prd,np.zeros(num_nodes),0)


#### Create gossip matrices
zero_row_sum,zero_col_sum,row_stochastic,col_stochastic, N_out, neighbors  = Random(num_nodes, prob=0.8, Laplacian_dividing_factor= 2).directed()


#### Run the optimization algorithms and compute the performance metrics 
x_dagp, _, _, _  = \
    dopt.DAGP(prd, zero_row_sum, zero_col_sum, learning_rate, max_iter_syn, \
                               theta_0, rho , alpha, cons = True)

x_asy_dagp, _, _, _, _, Delay_mat_dagp = \
    dopt.Asy_DAGP(T_active_exp1, Tv_nodes_exp1, prd, zero_row_sum, zero_col_sum, learning_rate, max_iter_asyn, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=False, drop_prob=0.)

f_DAGP     = error_prd.cost_path(np.sum(x_dagp,      axis=1)/num_nodes) 
f_asy_dagp = error_prd.cost_path(np.sum(x_asy_dagp,  axis=1)/num_nodes) 


##### part 2: experiments for the Throttled setup
np.random.seed(seed)

mincomp_2 = np.array([1,2,1,1,1,1,2,2,1,1])
maxcomp_2 = np.array([5,20,15,20,50,60,35,40,45,50])

T_active_exp1_2, Tv_nodes_exp1_2, node_comp_time_exp1_2 = \
    ut.create_computation_time(num_nodes, max_iter=int(1e5), comp_time_dist=comp_time_dist, mean_comp=None,\
                                min_comp = mincomp_2, max_comp=maxcomp_2, variance_comp=None, make_integer=True) 


x_dagp_2, _, _, _  = \
    dopt.DAGP(prd, zero_row_sum, zero_col_sum, learning_rate, max_iter_syn, \
                               theta_0, rho , alpha, cons = True)

x_asy_dagp_2, _, _, _, _, Delay_mat_dagp_2 = \
    dopt.Asy_DAGP(T_active_exp1_2, Tv_nodes_exp1_2,  prd, zero_row_sum, zero_col_sum, learning_rate, max_iter_asyn, num_nodes, dim, rho, alpha, gamma, eta, neighbors, \
             cons = True, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, \
                drop_msg=False, drop_prob=0.)

f_DAGP_2     = error_prd.cost_path(np.sum(x_dagp_2,      axis=1)/num_nodes) 
f_asy_dagp_2 = error_prd.cost_path(np.sum(x_asy_dagp_2,  axis=1)/num_nodes) 


#### save data and plot results
plot_exp1(f_DAGP, f_DAGP_2, f_asy_dagp, f_asy_dagp_2, max_iter_syn, node_comp_time_exp1, \
           node_comp_time_exp1_2, neighbors, Delay_mat_dagp, Delay_mat_dagp_2, T_active_exp1, \
            T_active_exp1_2, current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder='exp1')

