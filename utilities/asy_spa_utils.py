
import numpy as np
import copy as cp


def update_estimates_spa(node, Message, dim, current, w, y, l_tilde, neighbors):
    w_new   = np.zeros(dim)
    y_new   = 0
    l_list  = []
    for sender in np.where(neighbors[node]==1.)[0]:
        msg_val = Message[node][sender]
        msg_new = []
        count=0
        for val in msg_val:
            if val[0]<current:
                count+=1
                w_new+=val[1]
                y_new+=val[2]
                l_list.append(val[3])
            else:
                msg_new.append(val)
            if count>0:
                Message[node][sender] = msg_new

    w[node] = w_new
    y[node] = y_new
    l_tilde[node] = np.max(l_list)

    return Message, w, y, l_tilde

def broadcast_spa(node, x, y, l, current, Delay_mat, Message, itr, N_out, neighbors, dropping_msg, dropping_prob):
    x_send = x[node]/N_out[node]
    y_send = y[node]/N_out[node]
    l_send = l[node]
    counter = 0
    for receiver in np.where(neighbors[:,node]==1.)[0]:
        if receiver==node:
            Message[receiver][node].append([current,x_send,y_send,l_send])
        else:
            if dropping_msg:
                drop = np.random.binomial(1,dropping_prob)
                if drop == 1.:
                    counter += 1
                else:
                    receive_time=current+Delay_mat[receiver,node,itr]
                    Message[receiver][node].append([receive_time,x_send,y_send,l_send])
            else:
                receive_time=current+Delay_mat[receiver,node,itr]
                Message[receiver][node].append([receive_time,x_send,y_send,l_send])
    return Message, counter

def update_variables_spa(prd,node,z,x,w,y,f_grad,l,l_tilde,itr,learning_rate,correct_step_size,decreasing_step_size):
    z[node] = w[node]/y[node]
    f_grad[node] = prd.localgrad(z,node)
    if correct_step_size:
        if decreasing_step_size:
            tmp = l_tilde[node]-l[node]
            acc_step = 0
            for k in range(int(tmp+1)): ## one should be here
                acc_step += 1/(np.sqrt(l[node]+k))
            x[node] = w[node] - acc_step*learning_rate*(f_grad[node])
        else:
            x[node] = w[node] - (l_tilde[node]-l[node]+1)*learning_rate*f_grad[node]
    else:
        if decreasing_step_size:
            x[node] = w[node] - (1/np.sqrt(itr))*1*(f_grad[node])
        else:
            x[node] = w[node] - learning_rate*f_grad[node]
    l[node] = l_tilde[node]+1
    return z,x,w,y,f_grad,l

def save_vars_spa(x,z,w,l,y, x_out, z_out, w_out, l_out, y_out):
    x_out.append(cp.deepcopy(x))
    z_out.append(cp.deepcopy(z))
    # w_out.append(cp.deepcopy(w))
    # l_out.append(cp.deepcopy(l))
    # y_out.append(cp.deepcopy(y))
    return x_out, z_out, w_out, l_out, y_out

def Asy_SPA_init(prd, num_nodes, dim, max_iter, min_delay, max_delay, expScale_delay, delay_type,  N_out, neighbors, drop_msg=False, drop_prob=0.0001):

    # x = np.random.randn(num_nodes,dim) 
    x = np.zeros((num_nodes,dim))
    Message=[[[] for a in range(num_nodes)] for a in range(num_nodes)]


    f_grad = prd.networkgrad( x )
    y = np.ones(num_nodes)
    l = np.ones(num_nodes)
    z = cp.deepcopy(x)
    w = cp.deepcopy(x)

    if delay_type == 'uniform':
        Delay_mat = np.random.uniform(min_delay, max_delay, size=(num_nodes,num_nodes,max_iter+1))
    elif delay_type == 'exp':
        Delay_mat = np.random.exponential(expScale_delay,   size=(num_nodes,num_nodes,max_iter+1))

    active_set= []

    for node in range(num_nodes):
        Message, _ = broadcast_spa(node, x, y, l, current=0, Delay_mat=Delay_mat, Message=Message, itr=0, N_out=N_out, neighbors=neighbors,\
                                dropping_msg=drop_msg, dropping_prob=drop_prob)

    x_out = [cp.deepcopy(x)]
    y_out = [cp.deepcopy(y)]
    w_out = [cp.deepcopy(w)]
    z_out = [cp.deepcopy(z)]
    l_out = [cp.deepcopy(l)]

    return x, z, y, l, w, Message, Delay_mat, f_grad, active_set, x_out, z_out, y_out, w_out, l_out