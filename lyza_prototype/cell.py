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

            N_reshaped = np.hstack(N).reshape(-1,self.n_node)
            B_reshaped = np.hstack(B).reshape(-1,self.n_node)

            N_arr.append(N_reshaped)
            B_arr.append(B_reshaped)
            # import ipdb; ipdb.set_trace()


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
            for i in range(spatial_dim):
                for j in range(self.elem_dim):
                    J[i,j] += self.nodes[I].coor[i]*self.Bhat[I](xi)[j]

        return J


    # def interpolate_at_quad_point(self, function, quad_point_idx):
    #     result = np.zeros((function.function_size,1))

    #     for I in range(self.n_node):
    #         val = function.get_node_val(self.nodes[I].idx)
    #         # shape_function_val = self.N[I](self.quad_points[quad_point_idx].coor)

    #         for i in range(function.function_size):
    #             result[i] += self.quad_points[quad_point_idx].N[I]*val[i]

    #     return result

    # def interpolate_gradient_at_quad_point(self, function, quad_point_idx):
    #     result = np.zeros((function.function_size,self.spatial_dimension))

    #     for I in range(self.n_node):
    #         val = function.get_node_val(self.nodes[I].idx)
    #         # shape_function_val = self.N[I](self.quad_points[quad_point_idx].coor)
    #         for i in range(function.function_size):
    #             for j in range(self.spatial_dimension):
    #                 result[i,j] += self.quad_points[quad_point_idx].B[I][j]*val[i]

    #     # import ipdb; ipdb.set_trace()

    #     return result
