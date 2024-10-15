
import numpy as np
import copy as cp


def update_estimates_dagp(node, Message, A, B, dim, current, neighbors):
    for sender in np.where(neighbors[node]==1.)[0]:
        message_val=Message[node][sender]
        message_new=[]
        count=0
        a_new=np.zeros(dim)
        b_new=np.zeros(dim)
        for val in message_val:
            if val[0]<current:
                count+=1
                a_new+=val[1]
                b_new+=val[2]
            else:
                message_new.append(val)
        if count>0:
            Message[node][sender]=message_new
            A[node, sender,:]=a_new/count
            B[node, sender,:]=b_new/count
    return Message, A, B

def broadcast_dagp(node, x, p, current, Delay_mat, Message, itr, dropping_msg, dropping_prob, neighbors):
    x_send = x[node,:]
    p_send = p[node,:]
    counter = 0
    for receiver in np.where(neighbors[:,node]==1.)[0]:
        if receiver==node:
            Message[receiver][node].append([current,x_send,p_send])
        else:
            if dropping_msg:
                drop = np.random.binomial(1,dropping_prob)
                if drop == 1.:
                    counter += 1
                else:
                    receive_time=current+Delay_mat[receiver,node,itr]
                    Message[receiver][node].append([receive_time,x_send,p_send])
            else:
                receive_time=current+Delay_mat[receiver,node,itr]
                Message[receiver][node].append([receive_time,x_send,p_send])
    return Message, counter

def update_variables_dagp(node,prd,z,x,g,p,h,f_grad, R, learning_rate, alpha, eta, alpha_vr, rho, C, A, B, cons):
    z[node] = x[node] - np.matmul(R[node],A[node,:,:]) + learning_rate*(g[node] - f_grad[node])
    if cons:
        x[node] = prd.local_projection(node, z)
    else: 
        x[node] = cp.deepcopy(z[node])
    g[node] = g[node] + rho*(f_grad[node]-g[node]+ (z[node]-x[node])/(learning_rate)) + alpha*(h[node]-g[node])
    tmp = np.matmul(C[node], B[node,:,:])
    p[node] = p[node] - eta*tmp + eta*(alpha_vr-1)*g[node]
    h[node] = alpha_vr*h[node] - tmp
    f_grad[node] = prd.localgrad(x,node)
    return z, x, g, p, h, f_grad 

def Asy_DAGP_init(prd, num_nodes, dim, max_iter, min_delay, max_delay, neighbors, expScale_delay, delay_type, cons=False, drop_msg=False, drop_prob=0.001):
    z = np.random.randn(num_nodes,dim) 
    if cons:
        x = prd.network_projection(z) 
    else:
        x = cp.deepcopy(z)

    Message=[[[] for a in range(num_nodes)] for aa in range(num_nodes)]

    f_grad = prd.networkgrad(x)
    g = np.zeros(f_grad.shape)
    h = np.zeros(g.shape)
    p = np.zeros(g.shape)

    if delay_type == 'uniform':
        Delay_mat = np.random.uniform(min_delay, max_delay, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'exp':
        Delay_mat = np.random.exponential(expScale_delay,   size=(num_nodes,num_nodes,max_iter+2))
        

    active_set= []

    A = np.zeros((num_nodes,num_nodes,dim))
    for node in range(num_nodes):
        A[node,node,:] = x[node,:]
    B = np.zeros((num_nodes,num_nodes,dim))

    for node in range(num_nodes):
        Message, _ = broadcast_dagp(node, x, p, current=0, Delay_mat=Delay_mat, Message=Message, itr=0, dropping_msg=drop_msg, dropping_prob=drop_prob, neighbors=neighbors)

    x_out = [cp.deepcopy(x)]
    z_out = [cp.deepcopy(z)]
    g_out = [cp.deepcopy(g)]
    h_out = [cp.deepcopy(h)]
    p_out = [cp.deepcopy(p)]

    return x, z,  Message, Delay_mat, f_grad, g, h, p, A, B, active_set, x_out, z_out, g_out, h_out, p_out

def save_vars_dagp(x,z,g,h,p, x_out, z_out, g_out, h_out, p_out):
    x_out.append(cp.deepcopy(x))
    # z_out.append(cp.deepcopy(z))
    # g_out.append(cp.deepcopy(g))
    # h_out.append(cp.deepcopy(h))
    # p_out.append(cp.deepcopy(p))
    return x_out, z_out, g_out, h_out, p_out