import numpy as np
import networkx as nx

class Random:
    def __init__(self,number_of_nodes, prob, Laplacian_dividing_factor):
        self.size = number_of_nodes
        self.prob = prob
        self.LDF  = Laplacian_dividing_factor

    def directed(self):     # I should think more about how I am generating them. num_edges= N(N-1)
        nn = self.size*(self.size)
        indices = np.arange(nn)
        np.random.shuffle(indices)
        nonz = np.int(np.floor(nn*self.prob))
        ind = indices[:nonz]
        Z = np.zeros(nn)
        Z[ind] = 1.0
        D = Z.reshape(self.size,self.size)

        for i in range(self.size):
            if D[i][i] == 1.:
                D[i][i] = 0.

        GG = nx.from_numpy_matrix(np.matrix(D), create_using=nx.DiGraph)
        largest = max(nx.kosaraju_strongly_connected_components(GG), key=len)

        adj = np.zeros((len(largest), len(largest)))
        v = 0
        w = 0
        for i in largest:
            for j in largest:
                adj[v][w] = D[i][j]
                w +=1
            w = 0
            v +=1
        row_sum = np.sum(adj, axis = 1)
        col_sum = np.sum(adj, axis = 0)
        l_in  = np.diag(row_sum) - adj
        l_out = np.diag(col_sum) - adj
        ZR  = l_in  / (self.LDF*np.max(row_sum))
        ZC  = l_out / (self.LDF*np.max(col_sum))
        RS  = np.eye(self.size) - ZR
        CS  = np.eye(self.size) - ZC
        N_out = np.count_nonzero(CS,axis=0)
        neighbors = 1*(RS>0)
        return ZR, ZC, RS, CS, N_out, neighbors

    def undirected(self):     # I think I were wrong N(N-1)/2 edges can an undir_graph have
        nn = self.size*self.size
        indices = np.arange(nn)
        np.random.shuffle(indices)
        nonz = np.int(np.floor(nn*self.prob))
        ind = indices[:nonz]
        Z = np.zeros(nn)
        Z[ind] = 1.0
        U = Z.reshape(self.size,self.size)
        for i in range(self.size):
            if U[i][i] == 1.:
                U[i][i] = 0.
            for j in range(self.size):
                if U[i][j] == 1:
                    U[j][i] = 1
        adj = U
        row_sum = np.sum(adj, axis = 1)
        col_sum = np.sum(adj, axis = 0)
        l_in  = np.diag(row_sum) - adj
        l_out = np.diag(col_sum) - adj
        ZR  = l_in  / (self.LDF*np.max(row_sum))
        ZC  = l_out / (self.LDF*np.max(col_sum))
        RS  = np.eye(self.size) - ZR
        CS  = np.eye(self.size) - ZC
        return ZR, ZC, RS, CS
