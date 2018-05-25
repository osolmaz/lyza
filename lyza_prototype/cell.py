from math import sqrt, cos, sin, pi
import numpy as np
import itertools
from lyza_prototype.helper import determinant, inverse


class Cell:
    N = []
    Bhat = []
    elem_dim = None

    def __init__(self, nodes, is_boundary=False, label=None):
        self.label = label
        self.nodes = nodes
        self.n_node = len(self.nodes)
        self.is_boundary = is_boundary

    def all_nodes_in(self, position):
        return not (False in [position(node.coor) for node in self.nodes])

    def some_nodes_in(self, position):
        return True in [position(node.coor) for node in self.nodes]

    def get_quad_points(self, quadrature_degree):
        raise Exception('Do not use the base class')

    def calculate_basis_values(self, spatial_dim, quadrature_degree):
        self.n_node = len(self.nodes)

        quad_weights, quad_coors = self.get_quad_points(quadrature_degree)

        for i in quad_coors:
            if i.shape != (3,1):
                raise Exception('Invalid shape for quadrature point coordinates. Fix your element code.')

        N_arr = []
        B_arr = []
        jac_arr = []
        det_jac_arr = []
        jac_inv_tra_arr = []
        global_coor_arr = []

        for coor in quad_coors:
            jac = self.jacobian(coor, spatial_dim)
            det_jac = determinant(jac)
            jac_inv_tra = inverse(jac).transpose()

            jac_arr.append(jac)
            det_jac_arr.append(np.array(det_jac).reshape(1,1))
            jac_inv_tra_arr.append(jac_inv_tra)

            N = np.zeros((self.n_node,1))
            B = np.zeros((self.n_node,spatial_dim))

            for I in range(len(self.N)):
                N[I] = self.N[I](coor)
                B[I] = jac_inv_tra.dot(self.Bhat[I](coor)).T

            N_arr.append(N)
            B_arr.append(B)

            quad_point_global = [0. ,0., 0.]

            coor_matrix = np.zeros((self.n_node, 3))
            for idx, n in enumerate(self.nodes):
                coor_matrix[idx] = n.coor.T

            quad_point_global = coor_matrix.T.dot(N)

            # for I in itertools.product(range(self.n_node)):
            #     quad_point_global[i] += N[I]*self.nodes[I].coor[i]

            global_coor_arr.append(np.array(quad_point_global))

        return N_arr, B_arr, jac_arr, det_jac_arr, jac_inv_tra_arr, global_coor_arr

    def jacobian(self, xi, spatial_dim):
        J = np.zeros((spatial_dim,self.elem_dim))

        for I in range(len(self.nodes)):
            Bhat = self.Bhat[I](xi)
            coor = self.nodes[I].coor[:self.elem_dim]

            J += coor*Bhat.T

            # import ipdb; ipdb.set_trace()
            # for i in range(spatial_dim):
            #     for j in range(self.elem_dim):
            #         J[i,j] += coor[i]*Bhat[j]

        return J


