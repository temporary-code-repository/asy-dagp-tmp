
import numpy as np
import copy as cp


def broadcast_pg_extra(update_edges, node, x, y, current, Delay_mat, Message, itr, neighbors, dropping_msg, dropping_prob):
    x_send = x[node]
    y_send = y[update_edges[node],:]
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

def Asy_pg_extra_init(update_edges, prd, num_nodes, num_edges, dim, max_iter, min_delay, max_delay, neighbors, expScale_delay, delay_type, cons=False, drop_msg=False, drop_prob=0.001):
    z = np.random.randn(num_nodes,dim) 
    if cons:
        x = prd.network_projection(z) 
    else:
        x = cp.deepcopy(z)

    Message=[[[] for a in range(num_nodes)] for a in range(num_nodes)]

    f_grad = prd.networkgrad(x)

    if delay_type == 'uniform':
        Delay_mat = np.random.uniform(min_delay, max_delay, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'exp':
        Delay_mat = np.random.exponential(expScale_delay,   size=(num_nodes,num_nodes,max_iter+2))

    active_set= []

    A = np.zeros((num_nodes,num_nodes,dim))
    for node in range(num_nodes):
        A[node,node,:] = x[node,:]
    B = np.zeros((num_nodes,num_edges,dim))

    y = np.zeros((num_edges,dim))

    for node in range(num_nodes):
        Message, _ = broadcast_pg_extra(update_edges, node, x, y, current=0, Delay_mat=Delay_mat, Message=Message, itr=0, dropping_msg=drop_msg, dropping_prob=drop_prob, neighbors=neighbors)

    x_out = [cp.deepcopy(x)]
    z_out = [cp.deepcopy(z)]

    return x, z, y, Message, Delay_mat, f_grad, A, B, active_set, x_out, z_out

def update_vars_pg_extra(eta,update_edges,node,prd,z,x,y,f_grad,W,V,learning_rate,A,B,cons):
    x_pre = cp.deepcopy(x)
    y_pre = cp.deepcopy(y)
    z[node] = np.matmul(W[node],A[node,:,:]) - learning_rate*f_grad[node] - np.matmul(V[:,node],B[node,:,:])
    if cons:
        x[node] = prd.local_projection(node, z)
    else: 
        x[node] = cp.deepcopy(z[node])

    x[node] = (1-eta[node])*x_pre[node] + eta[node]*x[node]

    for edge in update_edges[node]:
        y[edge] = ( B[node,edge,:] + np.matmul(V[edge],A[node,:,:] ) )
        y[edge] = (1-eta[node])*y_pre[edge] + eta[node]*y[edge]
    f_grad[node] = prd.localgrad(x,node)
    return x, z, y, f_grad 

def find_pair_by_first_element(pairs, target):
    for pair in pairs:
        if pair[0] == target:
            return pair
    return None  

def update_estimates_pg_extra(update_edges, node, Message, A, B, dim, current, neighbors):
    for sender in np.where(neighbors[node]==1.)[0]:
        message_val=Message[node][sender]
        message_new=[]
        list_time = []
        for val in message_val:
            if val[0]<current:
                list_time.append(val[0])
            else:
                message_new.append(val)
        if len(list_time)>0:
            most_recent_message_received_time = max(list_time) 
            val = find_pair_by_first_element(message_val, most_recent_message_received_time)  
            Message[node][sender] = message_new
            A[node, sender,:] = val[1]
            B[node, update_edges[sender],:] = val[2]
    return Message, A, B
