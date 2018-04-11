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
        # self.stress = Quantity((6, 1), n_quad_point)
        # self.strain = Quantity((6, 1), n_quad_point)

        self.phi = Quantity((3, 1), n_quad_point)
        self.F = Quantity((3, 3), n_quad_point)


    # def calculate_stress(self, function):
    #     V = function.function_space
    #     voigt_index_map = [[0,3,5],[3,1,4],[5,4,2]]
    #     if V != self.elements[0].function_space:
    #         raise Exception('Function spaces do not match')

    #     for n_q, q in enumerate(self.elements[0].quad_points):
    #         grad_u = np.zeros((V.function_size, V.spatial_dimension))

    #         for I in range(self.elements[0].n_node):
    #             val = function.get_node_val(self.elements[0].nodes[I].idx)

    #             for i, j in itertools.product(range(V.function_size), range(V.spatial_dimension)):
    #                 grad_u[i, j] += val[i]*q.B[I][j]

    #         strain = (grad_u + grad_u.T)/2.
    #         stress = np.zeros((V.function_size, V.spatial_dimension))

    #         for i, j, k, l in itertools.product(
    #                 range(V.function_size), range(V.spatial_dimension),
    #                 range(V.function_size), range(V.spatial_dimension)):
    #             stress[i,j] += self.C[self.index_map[i][j], self.index_map[k][l]]*strain[k,l]

    #         strain_voigt = to_voigt(strain)
    #         stress_voigt = to_voigt(stress)

    #         self.stress.vectors[n_q] = stress_voigt
    #         self.strain.vectors[n_q] = strain_voigt
    #         # import ipdb; ipdb.set_trace()


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
            B = q.B

            # import ipdb; ipdb.set_trace()
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

            # for gamma,delta,a,b,A,B,C,D in itertools.product(
            #         range(n_node_1),
            #         range(n_node_2),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim)):

            #     alpha = gamma*spatial_dim + a
            #     beta = delta*spatial_dim + b
            #     C_val = self.C[self.index_map[A][B], self.index_map[C][D]]
            #     K[alpha, beta] += q.B[gamma][A]*C_val*E[C,D]*q.B[delta][B]\
            #                       *identity[a,b]*q.det_jac*q.weight


        # import ipdb; ipdb.set_trace()
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
        # self.stress = Quantity((6, 1), n_quad_point)
        # self.strain = Quantity((6, 1), n_quad_point)

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

            # import ipdb; ipdb.set_trace()
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

        # import ipdb; ipdb.set_trace()
        return f
