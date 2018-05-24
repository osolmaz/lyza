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

            B = []
            N = []
            for I in range(len(self.N)):
                N.append(self.N[I](coor))
                B.append(jac_inv_tra.dot(self.Bhat[I](coor)))

            N_reshaped = np.hstack(N).reshape(-1,self.n_node).T
            B_reshaped = np.hstack(B).reshape(-1,self.n_node).T

            N_arr.append(N_reshaped)
            B_arr.append(B_reshaped)

            quad_point_global = [0. ,0., 0.]
            for I, i in itertools.product(range(self.n_node), range(3)):
                quad_point_global[i] += N[I]*self.nodes[I].coor[i]

            global_coor_arr.append(np.array(quad_point_global))


        # self.quad_intp_matrix = np.zeros((self.n_quad_point, self.n_node))
        # for i, quad_point in enumerate(self.quad_points):
        #     for j, shape_function in enumerate(self.N):
        #         self.quad_intp_matrix[i,j] = shape_function(quad_point.coor)

        # self.quad_intp_matrix_inv = inverse(self.quad_intp_matrix)

        return N_arr, B_arr, jac_arr, det_jac_arr, jac_inv_tra_arr, global_coor_arr

    def jacobian(self, xi, spatial_dim):
        J = np.zeros((spatial_dim,self.elem_dim))

        for I in range(len(self.nodes)):
            Bhat = self.Bhat[I](xi)
            coor = self.nodes[I].coor[:spatial_dim]

            J += coor*Bhat

            # import ipdb; ipdb.set_trace()
            # for i in range(spatial_dim):
            #     for j in range(self.elem_dim):
            #         J[i,j] += coor[i]*Bhat[j]

        return J


