import numpy as np
from pylyza.quadrature_interface import MatrixQuadratureInterface
import itertools

def delta(i,j):
    if i==j:
        return 1
    else:
        return 0

class LinearElasticityMatrix(MatrixQuadratureInterface):


    def postinit(self):
        self.lambda_ = self.param['lambda']
        self.mu = self.param['mu']

    def C(self, i,j,k,l):
        return self.lambda_*delta(i,j)*delta(k,l) + self.mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))


    def eval(self, K, N_p, B_p, det_jac, physical_dim, elem_dim, n_dof, n_node):
        # K = np.zeros((n_dof,n_dof))
        # B_p = []
        # jac = self.jacobian(p)
        # det_jac = determinant(jac)
        # jac_inv_tra = inverse(jac).transpose()

        # for I in range(n_node):
        #     B_p.append(jac_inv_tra.dot(self.Bhat[I](p)))

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
