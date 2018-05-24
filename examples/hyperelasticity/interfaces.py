import numpy as np
from lyza_prototype.assembler import MatrixAssembler, VectorAssembler
from lyza_prototype.cell_iterator import CellIterator
import itertools

def st_venant_eulerian_elasticity(lambda_, mu, b_, a,b,c,d):
    return lambda_*b_[a,b]*b_[c,d] + mu*(b_[a,c]*b_[b,d]+b_[a,d]*b_[b,c])

class HyperelasticityJacobian(MatrixAssembler):

    def set_param(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        FINVT_arr = self.mesh.quantities['FINVT'].get_quantity(cell)
        BBAR_arr = self.mesh.quantities['BBAR'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        LCG_arr = self.mesh.quantities['LCG'].get_quantity(cell)
        TAU_arr = self.mesh.quantities['TAU'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        K = np.zeros((n_dof,n_dof))
        identity = np.eye(3)

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            for gamma,delta,a,b,c,d in itertools.product(
                    range(B.shape[0]),
                    range(B.shape[0]),
                    range(B.shape[1]),
                    range(B.shape[1]),
                    range(B.shape[1]),
                    range(B.shape[1])):

                alpha = gamma*B.shape[1] + a
                beta = delta*B.shape[1] + b
                c_val = st_venant_eulerian_elasticity(self.lambda_, self.mu, b_, a,c,b,d)
                K[alpha, beta] += Bbar[gamma][c]*c_val*Bbar[delta][d]*DETJ*W

            for gamma,delta,a,e,f in itertools.product(
                    range(B.shape[0]),
                    range(B.shape[0]),
                    range(B.shape[1]),
                    range(B.shape[1]),
                    range(B.shape[1])):

                alpha = gamma*B.shape[1] + a
                beta = delta*B.shape[1] + a
                K[alpha, beta] += Bbar[gamma][e]*tau[e,f]*Bbar[delta][f]*DETJ*W

        return K


class HyperelasticityResidual(VectorAssembler):

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        FINVT_arr = self.mesh.quantities['FINVT'].get_quantity(cell)
        BBAR_arr = self.mesh.quantities['BBAR'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        LCG_arr = self.mesh.quantities['LCG'].get_quantity(cell)
        TAU_arr = self.mesh.quantities['TAU'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        f = np.zeros((n_dof, 1))

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            for gamma,a,b in itertools.product(
                    range(B.shape[0]),
                    range(B.shape[1]),
                    range(B.shape[1])):

                alpha = gamma*B.shape[1] + a
                f[alpha] += -tau[a,b]*Bbar[gamma][b]*DETJ*W

        return f


# class CalculateStrainStress(ElementInterface):

#     def __init__(self, lambda_, mu):
#         self.lambda_ = lambda_
#         self.mu = mu

#     def init_quadrature_point_quantities(self, n_quad_point):
#         self.phi = Quantity((3, 1), n_quad_point)
#         self.F = Quantity((3, 3), n_quad_point)
#         self.E = Quantity((6, 1), n_quad_point)
#         self.S = Quantity((6, 1), n_quad_point)
#         self.sigma = Quantity((6, 1), n_quad_point)

#     def matrix(self):
#         n_dof = self.get_element_n_dofs()
#         n_node = self.get_element_n_nodes()
#         spatial_dim = self.elements[0].spatial_dimension

#         K = np.zeros(n_dof)

#         identity = np.eye(3)

#         for n, q in enumerate(self.elements[0].quad_points):
#             F = self.F.vectors[n]
#             E = 0.5 * (F.T.dot(F) - identity)
#             S = self.lambda_*np.trace(E)*np.identity(3) + 2*self.mu*E
#             b_ = F.dot(F.T)
#             tau = (self.lambda_/2*(np.trace(b_)-3)-self.mu)*b_ + self.mu*(b_.dot(b_))
#             sigma = tau/np.linalg.det(F)

#             self.S.vectors[n] = to_voigt(S)
#             self.E.vectors[n] = to_voigt(E)
#             self.sigma.vectors[n] = to_voigt(sigma)

#         return K


