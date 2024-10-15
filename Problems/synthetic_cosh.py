
import numpy as np



class synthetic(): ### minimize log(cosh(cx-e) s.t. ax-b<=0)
    def __init__(self, seed, num_nodes, dim):
        np.random.seed(seed)
        self.n = num_nodes
        self.d = dim
        self.A = np.random.randn(num_nodes, dim)
        self.e = np.random.randn(num_nodes)
        self.C = np.random.randn(num_nodes, dim)
        self.b = np.matmul(self.A, np.random.randn(dim)) + 1*(np.random.rand(num_nodes))

        self.N = 1
        self.X_train = None
        self.Y_train = None
        self.p = self.A.shape[1]
        self.dim = self.A.shape[1]

    def F_val(self, theta):
        return np.sum(np.log10(np.cosh(np.matmul(self.C,theta)-self.e)))

    def localgrad(self, theta, idx):
        grad2 = (1 / np.cosh(np.inner(self.C[idx],theta[idx])-self.e[idx])) *\
                np.sinh(np.inner(self.C[idx], theta[idx])-self.e[idx]) * self.C[idx]
        return grad2

    def gd_grad(self, theta, dim):
        grad = np.zeros(dim)
        for i in range(self.n):
            grad += (1 / np.cosh(np.inner(self.C[i],theta)-self.e[i])) *\
                np.sinh(np.inner(self.C[i], theta)-self.e[i]) * self.C[i]
        return grad

    def networkgrad(self, theta):
        grad = np.zeros((self.n, self.p))
        for i in range(self.n):
            grad[i] = self.localgrad(theta, i)
        return grad

    def grad(self, theta):
        pass

    def local_projection(self, idx, theta):
        tmp = np.inner(self.A[idx],theta[idx]) - self.b[idx]
        if tmp < 0:
            return theta[idx]
        else:
            return theta[idx] - tmp*( (self.A[idx]) / (np.linalg.norm(self.A[idx]))**2 )

    def network_projection(self, theta):
        proj = np.zeros((self.n, self.p))
        for i in range(self.n):
            proj[i] = self.local_projection(i, theta)
        return proj





