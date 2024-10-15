########################################################################################################################
####---------------------------------------------------Analysis-----------------------------------------------------####
########################################################################################################################
import numpy as np
from numpy import linalg as LA

class error:
    def __init__(self, problem, model_optimal, cost_optimal):
        self.pr = problem                                       ## problem class
        self.N = self.pr.N                                      ## total number of data samples
        self.X = self.pr.X_train                                ## feature vectors
        self.Y = self.pr.Y_train                                ## label vector
        self.theta_opt = model_optimal
        self.F_opt = cost_optimal

    def path_cls_error(self, iterates):
        iterates = np.array( iterates )
        Y_predict = np.matmul( self.X, iterates.T )
        error_matrix = np.multiply( Y_predict, self.Y[:,np.newaxis] ) < 0
        return np.sum( error_matrix, axis = 0 ) / self.N

    def point_cls_error(self, theta):
        Y_predict = np.matmul( self.X, theta )
        error = Y_predict * self.Y < 0
        return sum(error)/self.N

    def theta_gap_path(self, iterates):
        return np.apply_along_axis( LA.norm, 1, iterates - self.theta_opt ) **2

    def cost_gap_point(self, theta):
        return self.pr.F_val(theta) - self.F_opt

    def cost_gap_path(self, iterates):
        K = len(iterates)
        result = [ ]
        for k in range(K):
            result.append( error.cost_gap_point(self, iterates[k]) )
        return result

    def cost_point(self, theta):
        return self.pr.F_val(theta)

    def cost_path(self, iterates):
        K = len(iterates)
        result = [ ]
        for k in range(K):
            result.append( error.cost_point(self, iterates[k]) )
        return result

    def feasibility_gap_syn(self, iterates):
        K = iterates.shape[0]
        result = [ ]
        A = self.pr.A
        n = np.sum(A**2, axis=1)
        n = n**(1/2)
        for k in range(K):
            tmp = (np.matmul(A, iterates[k]) - self.pr.b )/ n
            tmp[tmp<0] = 0
            result.append(np.sum(tmp**2) )
        return result
