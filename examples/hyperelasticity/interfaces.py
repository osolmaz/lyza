import numpy as np
from lyza_prototype.element_interface import ElementInterface
from lyza_prototype.quantity import Quantity
import itertools


def to_voigt(matrix):
    result = np.zeros((6,1))
    voigt_index_map = [[0,3,5],[3,1,4],[5,4,2]]
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        result[voigt_index_map[i][j]] = matrix[i,j]
    return result


class LagrangianHyperElasticityTangent(ElementInterface):

    def __init__(self, lambda_, mu):
        self.C = np.array([
            [lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])

        self.index_map = [[0,3,5],[3,1,4],[5,4,2]]

    def init_quadrature_point_quantities(self, n_quad_point):
        self.phi = Quantity((3, 1), n_quad_point)
        self.F = Quantity((3, 3), n_quad_point)

    def matrix(self):
        n_node_1 = len(self.elements[0].nodes)
        n_node_2 = len(self.elements[1].nodes)

        n_dof_1 = n_node_1*self.elements[0].function_size
        n_dof_2 = n_node_2*self.elements[1].function_size

        spatial_dim = self.elements[0].spatial_dimension

        K = np.zeros((n_dof_2,n_dof_1))
        identity = np.eye(3)

        for n, q in enumerate(self.elements[0].quad_points):
            F = self.F.vectors[n]
            E = 0.5 * (F.T.dot(F) - identity)

            for gamma,delta,a,b,A,B,C,D in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                beta = delta*spatial_dim + b
                C_val = self.C[self.index_map[A][B], self.index_map[C][D]]
                K[alpha, beta] += F[a,A]*q.B[gamma][B]*C_val \
                                  *F[b,C]*q.B[delta][D]*q.det_jac*q.weight

            for gamma,delta,a,A,B,C,D in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                beta = delta*spatial_dim + a
                C_val = self.C[self.index_map[A][B], self.index_map[C][D]]
                K[alpha, beta] += q.B[gamma][A]*C_val*E[C,D]*q.B[delta][B]\
                                  *q.det_jac*q.weight

        return K


class LagrangianHyperElasticityResidual(ElementInterface):

    def __init__(self, lambda_, mu):
        self.C = np.array([
            [lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])

        self.index_map = [[0,3,5],[3,1,4],[5,4,2]]

    def init_quadrature_point_quantities(self, n_quad_point):
        self.phi = Quantity((3, 1), n_quad_point)
        self.F = Quantity((3, 3), n_quad_point)

    def vector(self):
        n_dof = self.get_element_n_dofs()
        n_node = self.get_element_n_nodes()
        spatial_dim = self.elements[0].spatial_dimension

        f = np.zeros((n_dof, 1))

        identity = np.eye(3)

        for n, q in enumerate(self.elements[0].quad_points):
            F = self.F.vectors[n]
            E = 0.5 * (F.T.dot(F) - identity)

            for gamma,a,A,B,C,D in itertools.product(
                    range(n_node),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                C_val = self.C[self.index_map[A][B], self.index_map[C][D]]
                f[alpha] += -C_val*E[C,D]*F[a,A]*q.B[gamma][B]*q.det_jac*q.weight

        return f


def st_venant_eulerian_elasticity(lambda_, mu, b_, a,b,c,d):
    return lambda_*b_[a,b]*b_[c,d] + mu*(b_[a,c]*b_[b,d]+b_[a,d]*b_[b,c])


class EulerianHyperElasticityTangent(ElementInterface):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def init_quadrature_point_quantities(self, n_quad_point):
        self.phi = Quantity((3, 1), n_quad_point)
        self.F = Quantity((3, 3), n_quad_point)

    def matrix(self):
        n_node_1 = len(self.elements[0].nodes)
        n_node_2 = len(self.elements[1].nodes)

        n_dof_1 = n_node_1*self.elements[0].function_size
        n_dof_2 = n_node_2*self.elements[1].function_size

        spatial_dim = self.elements[0].spatial_dimension

        K = np.zeros((n_dof_2,n_dof_1))
        identity = np.eye(3)

        for n, q in enumerate(self.elements[0].quad_points):
            F = self.F.vectors[n]
            Finvtra = np.linalg.inv(F).T
            E = 0.5 * (F.T.dot(F) - identity)
            S = self.lambda_*np.trace(E)*np.identity(3) + 2*self.mu*E
            b_ = F.dot(F.T)
            # tau = F.dot(S).dot(F.T)
            tau = (self.lambda_/2*(np.trace(b_)-3)-self.mu)*b_ + self.mu*(b_.dot(b_))

            Bbar = []
            for B_I in q.B:
                Bbar.append(Finvtra.dot(B_I))
            # import ipdb; ipdb.set_trace()

            for gamma,delta,a,b,c,d in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                beta = delta*spatial_dim + b
                c_val = st_venant_eulerian_elasticity(self.lambda_, self.mu, b_, a,c,b,d)
                K[alpha, beta] += Bbar[gamma][c]*c_val*Bbar[delta][d]*q.det_jac*q.weight

            for gamma,delta,a,e,f in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                beta = delta*spatial_dim + a
                K[alpha, beta] += Bbar[gamma][e]*tau[e,f]*Bbar[delta][f]*q.det_jac*q.weight

        return K


class EulerianHyperElasticityResidual(ElementInterface):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def init_quadrature_point_quantities(self, n_quad_point):
        self.phi = Quantity((3, 1), n_quad_point)
        self.F = Quantity((3, 3), n_quad_point)

    def vector(self):
        n_dof = self.get_element_n_dofs()
        n_node = self.get_element_n_nodes()
        spatial_dim = self.elements[0].spatial_dimension

        f = np.zeros((n_dof, 1))

        identity = np.eye(3)

        for n, q in enumerate(self.elements[0].quad_points):
            F = self.F.vectors[n]
            Finvtra = np.linalg.inv(F).T
            E = 0.5 * (F.T.dot(F) - identity)
            S = self.lambda_*np.trace(E)*np.identity(3) + 2*self.mu*E
            b_ = F.dot(F.T)
            # tau = F.dot(S).dot(F.T)
            tau = (self.lambda_/2*(np.trace(b_)-3)-self.mu)*b_ + self.mu*(b_.dot(b_))

            Bbar = []
            for B_I in q.B:
                Bbar.append(Finvtra.dot(B_I))

            for gamma,a,b in itertools.product(
                    range(n_node),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = gamma*spatial_dim + a
                f[alpha] += -tau[a,b]*Bbar[gamma][b]*q.det_jac*q.weight

        return f

