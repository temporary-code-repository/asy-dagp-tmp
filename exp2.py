
import numpy as np
from analysis.analysis import error
from graph.graph import Random
from Optimizers import DOPTIMIZER as dopt
from utilities import utilities as ut
from Problems.logistic_regression import LR_L2
from Optimizers import COPTIMIZER as copt
import os
from utilities.plot_utils import plot_exp2


seed       = np.random.randint(12345)  
seed       = 8075
np.random.seed(seed)


#### create asynchronous setup 
num_nodes  = 20
mean_comp  = 1/np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

T_active_exp2, Tv_nodes_exp2, node_comp_time_exp2 = \
    ut.create_computation_time(num_nodes, max_iter=int(12000), comp_time_dist='exp', mean_comp=mean_comp,\
                                min_comp = None, max_comp=None, variance_comp=None, make_integer=True) 


#### Logistic Regression Problem ==> p: dimension of the model,   L: L-smooth constant,  N: total number of training samples,  b: average number of local samples,  l: regularization factor  
lr_0 = LR_L2(num_nodes, limited_labels=False, balanced=True, train=1000 , regularization=True, lamda=None)  

#### Create gossip matrices
zero_row_sum, zero_column_sum, row_stochastic, col_stochastic, N_out, neighbors  = Random(num_nodes, prob=0.6, Laplacian_dividing_factor= 2).directed()

#### some parameters of the algorithms
depoch    = 12500
rho       = 0.1
alpha     = 0.7
gamma     = 0.5
eta       = 1.0
expScale  = 1/50

step_asy_dagp   = 1.5/lr_0.L
step_asy_spa    = 1/lr_0.L/12  
step_center     = 1/lr_0.L/2
step_size_appg  = 1.5/lr_0.L


## find the optimal solution of Logistic regression
theta_c0    = np.random.normal(0,1,lr_0.p)
cepoch      = 60000
_, theta_opt, F_opt = copt.CGD(lr_0,step_center, cepoch, theta_c0)
error_lr_0 = error(lr_0,theta_opt,F_opt)


#### Run the optimization algorithms and compute the performance metrics
x_asy_dagp, _, _, _, _, _ = \
    dopt.Asy_DAGP(T_active_exp2, Tv_nodes_exp2,  lr_0, zero_row_sum, zero_column_sum, step_asy_dagp, depoch, num_nodes, lr_0.p, rho, alpha, gamma, eta, neighbors, \
                    cons=False, delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, drop_msg=False, drop_prob=0.)

x_asyspa, _, _, _ = \
    dopt.Asy_SPA(T_active_exp2, Tv_nodes_exp2, lr_0, step_asy_spa, depoch, num_nodes, lr_0.p, N_out, neighbors, delay_type='exp', min_delay=None, max_delay=None,\
                   expScale_delay=expScale, decreasing_step_size=False, correct_step_size=True, drop_msg=False, drop_prob=0.)

x_appg, _, _, _ = \
    dopt.APPG(T_active_exp2, Tv_nodes_exp2, lr_0, step_size_appg, depoch, num_nodes, lr_0.p,  N_out, neighbors, \
                delay_type='exp', min_delay=None, max_delay=None, expScale_delay=expScale, drop_msg=False, drop_prob=0.)


res_F_asy_dagp = error_lr_0.cost_gap_path( np.sum(x_asy_dagp, axis = 1)/num_nodes)
res_F_asyspa   = error_lr_0.cost_gap_path( np.sum(x_asyspa,   axis = 1)/num_nodes)
res_F_appg     = error_lr_0.cost_gap_path( np.sum(x_appg,     axis = 1)/num_nodes)


#### save data and plot results
plot_exp2(T_active_exp2, res_F_asy_dagp, res_F_asyspa, res_F_appg, \
           current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder='exp2')
