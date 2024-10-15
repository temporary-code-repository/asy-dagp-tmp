################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
from utilities import utilities as ut
from numpy import linalg as LA
from utilities.pg_extra_utils import *
from utilities.asy_dagp_utils import *
from utilities.asy_spa_utils import *
from utilities.appg_utils import *

def GP(prd,B,learning_rate,K,theta_0):
    theta = [cp.deepcopy( theta_0 )]
    grad = prd.networkgrad( theta[-1] )
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta.append( np.matmul( B, theta[-1] ) - learning_rate * grad )
        Y = np.matmul( B, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta[-1] )
        grad = prd.networkgrad( z )
        ut.monitor('GP', k, K)
    return theta


def ADDOPT(prd,B1,B2,learning_rate,K,theta_0):
    theta = [ cp.deepcopy(theta_0) ]
    grad = prd.networkgrad( theta[-1] )
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta.append( np.matmul( B1, theta[-1] ) - learning_rate * tracker )
        grad_last = cp.deepcopy(grad)
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta[-1] )
        grad = prd.networkgrad( z )
        tracker = np.matmul( B2, tracker ) + grad - grad_last
        ut.monitor('ADDOPT', k ,K)
    return theta


def SGP(prd,B,learning_rate,K,theta_0):
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad( theta, sample_vec )
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta = np.matmul( B, theta ) - learning_rate * grad
        Y = np.matmul( B, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad = prd.networkgrad( z, sample_vec )
        ut.monitor('SGP', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )
    return theta_epoch


def SADDOPT(prd,B1,B2,learning_rate,K,theta_0):
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad( theta, sample_vec )
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta = np.matmul( B1, theta ) - learning_rate * tracker
        grad_last = cp.deepcopy(grad)
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad = prd.networkgrad( z, sample_vec )
        tracker = np.matmul( B2, tracker ) + grad - grad_last
        ut.monitor('SADDOPT', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )
    return theta_epoch


def PushPull(prd,R,C,learning_rate,K,theta_0):
    theta = [ cp.deepcopy(theta_0) ]
    g = prd.networkgrad( theta[-1] )
    last_g = np.copy(g)
    y = np.copy(g)
    for k in range(K):
        theta.append( np.matmul( R, theta[-1]-learning_rate*y ) )
        last_g  = g
        g =  prd.networkgrad( theta[-1] )
        y = np.matmul( C, y ) + g - last_g
        ut.monitor('PushPull', k, K)
    return theta


def DAGP(prd, W, Q, learning_rate, K, x0, rho, alpha, cons = True):
    x = [cp.deepcopy(x0)]
    z = [cp.deepcopy(x0)]

    f_grad = prd.networkgrad(x[-1])
    g       = np.zeros(f_grad.shape)
    h       = np.zeros(g.shape)

    h_iterates = [cp.deepcopy(h)]
    g_iterates = [cp.deepcopy(g)]

    for k in range(K):
        z.append(x[-1] - np.matmul(W, x[-1])  + learning_rate*(g-f_grad))
        if cons:
            x.append(prd.network_projection( z[-1]))
        else:
            x.append(z[-1])
        local_grad = prd.networkgrad(x[-1])
        new_h = h - np.matmul(Q, h-g)
        g = g + rho*(f_grad-g + (z[-1]-x[-1])/learning_rate) + alpha*(h-g)
        f_grad = local_grad
        h = new_h
        h_iterates.append(h)
        g_iterates.append(g)
        ut.monitor('DAGP', k, K)
    return x, z, h_iterates, g_iterates


def APPG(T_active, Tv_nodes, prd, learning_rate, max_iter, num_nodes, dim,  N_out, Neighbors, \
        delay_type='uniform', min_delay=0, max_delay=10, expScale_delay=10, drop_msg=False, drop_prob=0.001):
    
    x, g, y, Message, Delay_mat, f_grad, active_set, x_out, g_out, y_out = \
        APPG_init(prd, num_nodes, dim, max_iter=max_iter,  N_out=N_out, Neighbors=Neighbors, min_delay=min_delay, max_delay=max_delay, \
                expScale_delay=expScale_delay, delay_type=delay_type, drop_msg=drop_msg, drop_prob=drop_prob)
    
    itr = 0
    drop_counter = 0
    x_avg = cp.deepcopy(x)
    y_sum = cp.deepcopy(y)

    for t_active in T_active[1:max_iter]:
        itr += 1
        active_set.append(ut.active_set(t_active, Tv_nodes))

        for node in active_set[-1]:
            Message, x_avg, y_sum = update_estimates_appg(node, Message, dim, t_active, x_avg, y_sum, Neighbors)

        for node in active_set[-1]:
            x, y, g               = update_variables_appg(prd, node, x_avg, y_sum, g, learning_rate, x, y)

        for node in active_set[-1]:
            Message, tmp_counter  = broadcast_appg(node, x, y, t_active, Delay_mat, Message, itr, N_out, Neighbors, dropping_msg=drop_msg, dropping_prob=drop_prob)
            drop_counter += tmp_counter

        x_out, y_out, g_out = save_vars_appg(x, y, g, x_out, y_out, g_out)

        ut.monitor('APPG',itr,max_iter)

    if drop_msg:
        print(f'number of dropped messages: {drop_counter}')

    return x_out, y_out, g_out, Delay_mat


def Asy_DAGP(T_active, Tv_nodes, prd, R, C, learning_rate, max_iter, num_nodes, dim, rho, alpha, alpha_vr, eta, neighbors, \
        cons = False, delay_type='uniform', min_delay=0, max_delay=10, expScale_delay=10, drop_msg=False, drop_prob=0.001):

    x, z, Message, Delay_mat, f_grad, g, h, p, A, B, active_set, x_out, z_out, g_out, h_out, p_out = Asy_DAGP_init(prd, num_nodes, dim, \
            max_iter=max_iter, min_delay=min_delay, max_delay=max_delay, neighbors=neighbors, expScale_delay=expScale_delay,\
                  delay_type=delay_type, cons=cons, drop_msg=drop_msg, drop_prob=drop_prob)

    drop_counter = 0
    itr = 0
    for t_active in T_active[1:max_iter]:
        itr += 1
        active_set.append(ut.active_set(t_active, Tv_nodes))
        
        for node in active_set[-1]:
            z, x, g, p, h, f_grad = update_variables_dagp(node,prd,z,x,g,p,h,f_grad,R,learning_rate,alpha,eta,alpha_vr,rho,C,A,B,cons)
            
        for node in active_set[-1]:
            Message, tmp_counter  = broadcast_dagp(node, x, p, t_active, Delay_mat, Message, itr, dropping_msg=drop_msg, dropping_prob=drop_prob, neighbors=neighbors)
            drop_counter += tmp_counter

        for node in active_set[-1]:
            Message, A, B         = update_estimates_dagp(node, Message, A, B, dim, t_active, neighbors)

        x_out, z_out, g_out, h_out, p_out = save_vars_dagp(x,z,g,h,p, x_out, z_out, g_out, h_out, p_out)
        ut.monitor('ASY-DAGP',itr,max_iter)
    if drop_msg:
        print(f'number of dropped messages: {drop_counter}')

    return x_out, z_out, g_out, h_out, p_out, Delay_mat


def Asy_SPA(T_active, Tv_nodes, prd, learning_rate, max_iter, num_nodes, dim, N_out, neighbors,  \
        delay_type, min_delay=0, max_delay=10,expScale_delay=10, decreasing_step_size=True, correct_step_size=True, drop_msg=False, drop_prob=0.001):

    x, z, y, l_tilde, w, Message, Delay_mat, f_grad, active_set, x_out, z_out, y_out, w_out, l_out = \
        Asy_SPA_init(prd, num_nodes, dim, max_iter=max_iter, min_delay=min_delay, max_delay=max_delay, expScale_delay=expScale_delay, delay_type=delay_type,\
                   N_out=N_out, neighbors=neighbors, drop_msg=drop_msg, drop_prob=drop_prob )

    itr = 0
    drop_counter = 0
    l = np.ones(num_nodes)

    for t_active in T_active[1:max_iter]:
        itr += 1
        active_set.append(ut.active_set(t_active, Tv_nodes))

        for node in active_set[-1]:
            Message, w, y, l_tilde        = update_estimates_spa(node, Message, dim, t_active, w, y, l_tilde, neighbors)

        for node in active_set[-1]:
            z,x,w,y,f_grad,l              = update_variables_spa(prd,node,z,x,w,y,f_grad,l,l_tilde,itr,\
                                                          learning_rate,correct_step_size=correct_step_size,decreasing_step_size=decreasing_step_size)

        for node in active_set[-1]:
            Message, tmp_counter          = broadcast_spa(node, x, y, l, t_active, Delay_mat, Message, itr, N_out, neighbors, dropping_msg=drop_msg, dropping_prob=drop_prob)
            drop_counter += tmp_counter

        x_out, z_out, w_out, l_out, y_out = save_vars_spa(x, z, w, l, y, x_out, z_out, w_out, l_out, y_out)

        ut.monitor('ASY-SPA',itr,max_iter)

    if drop_msg:
        print(f'number of dropped messages: {drop_counter}')
 
    return x_out, w_out, z_out, Delay_mat



def Asy_pg_extra(eta, T_active, Tv_nodes, prd, W, V, update_edges, learning_rate, max_iter, num_nodes, num_edges, dim, neighbors, \
        cons = True, delay_type='uniform', min_delay=0, max_delay=10, expScale_delay=10, drop_msg=False, drop_prob=0.000):

    x, z, y, Message, Delay_mat, f_grad, A, B, active_set, x_out, _ = Asy_pg_extra_init(update_edges, prd, num_nodes, num_edges, dim, \
            max_iter=max_iter, min_delay=min_delay, max_delay=max_delay, neighbors=neighbors, expScale_delay=expScale_delay,\
                  delay_type=delay_type, cons=cons, drop_msg=drop_msg, drop_prob=drop_prob)

    drop_counter = 0
    itr = 0
    for t_active in T_active[1:max_iter]:
        itr += 1
        active_set.append(ut.active_set_continuous(num_nodes, t_active, Tv_nodes, threshold=1e-7))
            
        for node in active_set[-1]:
            x, z, y, f_grad       = update_vars_pg_extra(eta, update_edges,node,prd,z,x,y,f_grad,W,V,learning_rate,A,B,cons)
        
        for node in active_set[-1]:
            Message, tmp_counter  = broadcast_pg_extra(update_edges, node, x, y, t_active, Delay_mat, Message, itr, dropping_msg=drop_msg, dropping_prob=drop_prob, neighbors=neighbors)
            drop_counter += tmp_counter

        for node in active_set[-1]:
            Message, A, B         = update_estimates_pg_extra(update_edges, node, Message, A, B, dim, t_active, neighbors)
            
        x_out.append(cp.deepcopy(x))
        ut.monitor('ASY-PG-EXTRA',itr,max_iter)
    if drop_msg:
        print(f'number of dropped messages: {drop_counter}')

    return x_out, Delay_mat, active_set


def Sayed_sync(prd, W, V, learning_rate, K, x0, y0):
    x = [cp.deepcopy(x0)]
    z = [cp.deepcopy(x0)]
    y = [cp.deepcopy(y0)]
    f_grad = prd.networkgrad( x[-1] )

    for k in range(K):
        z.append( np.matmul(W,x[-1]) - learning_rate*f_grad - np.matmul(np.transpose(V),y[-1]) )
        x.append( prd.network_projection(z[-1]) )
        y.append( y[-1] + np.matmul(V,x[-2] ) )

        f_grad = prd.networkgrad( x[-1] )
        ut.monitor('Sayed\'s synchronous', k, K)
    return x, y


def pg_extra(prd, W, W_tilde, learning_rate, K, x0):
    threshold = learning_rate*0.1
    x = [cp.deepcopy(x0)]
    f_grad_pre = prd.networkgrad(x[-1])
    x_intermidiate = [np.matmul(W,x0) - learning_rate*(f_grad_pre)]
    x.append(prd.network_projection(x_intermidiate[-1]))
    f_grad = prd.networkgrad(x[-1])

    for k in range(K):
        tmp1 = np.matmul(W,x[-1]) + x_intermidiate[-1] - np.matmul(W_tilde,x[-2])
        tmp2 = learning_rate*( f_grad - f_grad_pre ) 
        x_intermidiate.append(tmp1 - tmp2)
        x.append( prd.network_projection(x_intermidiate[-1]) )
        f_grad_pre = cp.deepcopy(f_grad)
        f_grad = prd.networkgrad( x[-1] )

        ut.monitor('PG-EXTRA\'s synchronous', k, K)
    return x


def p_extra(prd, W, W_tilde, learning_rate, K, x0):
    x = [cp.deepcopy(x0)]
    x_intermidiate = [np.matmul(W,x0)]
    x.append(prd.network_projection(x_intermidiate[-1]))

    for k in range(K):
        x_intermidiate.append(np.matmul(W,x[-1]) + x_intermidiate[-1] - np.matmul(W_tilde,x[-2]))
        x.append( prd.network_projection(x_intermidiate[-1]) )

        ut.monitor('P-EXTRA\'s synchronous', k, K)
    return x