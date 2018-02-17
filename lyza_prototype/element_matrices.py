import numpy as np
from lyza_prototype.element_interface import BilinearElementInterface
import itertools

def delta(i,j):
    if i==j:
        return 1
    else:
        return 0

class PoissonMatrix(BilinearElementInterface):

    def matrix(self):

        K = np.zeros((self.elem2.n_dof, self.elem1.n_dof))

        for q1, q2 in zip(self.elem1.quad_points, self.elem2.quad_points):

            for I,J,i in itertools.product(
                    range(self.elem1.n_node),
                    range(self.elem2.n_node),
                    range(self.elem1.physical_dimension)):

                K[I, J] += q1.B[I][i]*q2.B[J][i]*q1.det_jac*q1.weight

        return K

class LinearElasticityMatrix(BilinearElementInterface):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def C(self, i,j,k,l):
        return self.lambda_*delta(i,j)*delta(k,l) + self.mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))


    def matrix(self):
        n_node_1 = len(self.elem1.nodes)
        n_node_2 = len(self.elem2.nodes)

        n_dof_1 = n_node_1*self.elem1.function_dimension
        n_dof_2 = n_node_2*self.elem2.function_dimension

        physical_dim = self.elem1.physical_dimension

        K = np.zeros((n_dof_2,n_dof_1))

        for q1, q2 in zip(self.elem1.quad_points, self.elem2.quad_points):

            for I,J,i,j,k,l in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(physical_dim),
                    range(physical_dim),
                    range(physical_dim),
                    range(physical_dim)):

                alpha = I*physical_dim + i
                beta = J*physical_dim + j
                K[alpha, beta] += q1.B[I][k]*self.C(i,k,j,l)*q2.B[J][l]*q1.det_jac*q1.weight


        return K

