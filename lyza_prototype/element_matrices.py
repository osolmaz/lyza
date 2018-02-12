import numpy as np
from lyza_prototype.quadrature_interface import ElementMatrix
import itertools

def delta(i,j):
    if i==j:
        return 1
    else:
        return 0

class LinearElasticityMatrix(ElementMatrix):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu
        super().__init__()

    def C(self, i,j,k,l):
        return self.lambda_*delta(i,j)*delta(k,l) + self.mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))


    def eval(self, K, N_p, B_p, det_jac, quad_point, function_dim, physical_dim, elem_dim, n_dof, n_node):

        for I,J,i,j,k,l in itertools.product(
                range(n_node),
                range(n_node),
                range(physical_dim),
                range(physical_dim),
                range(physical_dim),
                range(physical_dim)):

            alpha = I*physical_dim + i
            beta = J*physical_dim + j
            K[alpha, beta] += B_p[I][k]*self.C(i,k,j,l)*B_p[J][l]*det_jac

        return K
