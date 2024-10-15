
import numpy as np
import copy as cp


def update_estimates_appg(node, Message, dim, current, X_avg, Y_sum, neighbors):
    x_sum = np.zeros(dim)
    y_sum = np.zeros(dim)
    
    counter = 0
    for sender in np.where(neighbors[node]==1.)[0]:
        msg_val = Message[node][sender]
        msg_new = []
        count=0
        for val in msg_val:
            if val[0]<current:
                count+=1
                x_sum+=val[1]
                y_sum+=val[2]
            else:
                msg_new.append(val)
        if count>0:
            Message[node][sender] = msg_new
        counter += count

    X_avg[node] = x_sum/counter
    Y_sum[node] = y_sum

    return Message, X_avg, Y_sum

def broadcast_appg(node, x, y, current, Delay_mat, Message, itr, N_out, neighbors, dropping_msg, dropping_prob):
    x_send = x[node]
    y_send = y[node]/N_out[node]
    counter = 0
    for receiver in np.where(neighbors[:,node]==1.)[0]:
        if receiver==node:
            Message[receiver][node].append([current,x_send,y_send])
        else:
            if dropping_msg:
                drop = np.random.binomial(1,dropping_prob)
                if drop == 1.:
                    counter += 1
                else:
                    receive_time=current+Delay_mat[receiver,node,itr]
                    Message[receiver][node].append([receive_time,x_send,y_send])
            else:
                receive_time=current+Delay_mat[receiver,node,itr]
                Message[receiver][node].append([receive_time,x_send,y_send])
    return Message, counter

def update_variables_appg(prd, node, x_avg, y_sum, g, learning_rate, x, y):
    pre_g = cp.deepcopy(g)
    g_new = prd.localgrad(x_avg, node)

    y[node] = y_sum[node] + g_new - pre_g[node]
    x[node] = x_avg[node] - learning_rate*y[node]
    pre_g[node] = g_new
    
    return x, y, pre_g

def APPG_init(prd, num_nodes, dim, max_iter,  N_out, Neighbors, \
              min_delay, max_delay, expScale_delay, delay_type, drop_msg=False, drop_prob=0.000):
    
    x = np.random.randn(num_nodes,dim) 
    x = np.zeros((num_nodes,dim))

    Message=[[[] for a in range(num_nodes)] for a in range(num_nodes)]

    f_grad = prd.networkgrad(x)
    g = cp.deepcopy(f_grad)
    y = cp.deepcopy(f_grad)

    if delay_type == 'uniform':
        Delay_mat = np.random.uniform(min_delay, max_delay, size=(num_nodes,num_nodes,max_iter+1))
    elif delay_type == 'exp':
        Delay_mat = np.random.exponential(expScale_delay,   size=(num_nodes,num_nodes,max_iter+1))

    active_set= []

    for node in range(num_nodes):
        Message,_ = broadcast_appg(node, x, y, current=0, Delay_mat=Delay_mat, Message=Message, \
                                    itr=0, N_out=N_out, neighbors=Neighbors, dropping_msg=drop_msg, dropping_prob=drop_prob)

    x_out = [cp.deepcopy(x)]
    g_out = [cp.deepcopy(g)]
    y_out = [cp.deepcopy(y)]

    return x, g, y,  Message, Delay_mat, f_grad, active_set, x_out, g_out, y_out

def save_vars_appg(x,y,g, x_out,y_out,g_out):
    x_out.append(cp.deepcopy(x))
    # y_out.append(cp.deepcopy(y))
    # g_out.append(cp.deepcopy(g))
    return x_out, y_out, g_out