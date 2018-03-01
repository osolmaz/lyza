from math import sqrt, cos, sin, pi
import numpy as np
import itertools
from lyza_prototype.helper import determinant, inverse



class FiniteElement:
    N = []
    Bhat = []
    elem_dim = None

    def __init__(self,
                 nodes,
                 parent_cell,
                 function_space,
                 quadrature_degree):

        self.function_space = function_space
        self.function_size = self.function_space.get_dimension()
        self.spatial_dimension = self.function_space.spatial_dimension
        self.quad_points = parent_cell.get_quad_points(quadrature_degree)
        self.n_quad_point = len(self.quad_points)

        self.nodes = nodes
        self.n_node = len(self.nodes)
        self.n_dof = self.n_node*self.function_size
        self.parent_cell = parent_cell

        self.dofmap = []
        for n in self.parent_cell.nodes:
            node_dofs = [n.idx*self.function_size+i for i in range(self.function_size)]
            self.dofmap += self.function_space.node_dofs[n.idx]

        if not self.N or not self.Bhat:
            raise Exception('Improper element subclassing')

        self.n_node = len(self.nodes)

        for quad_point in self.quad_points:
            quad_point.set_jacobian(self.jacobian(quad_point.coor))

            B = []
            N = []
            for I in range(len(self.N)):
                N.append(self.N[I](quad_point.coor))
                B.append(quad_point.jac_inv_tra.dot(self.Bhat[I](quad_point.coor)))

            quad_point.set_shape_function(N, B)

            quad_point_global = [0. ,0., 0.]
            for I, i in itertools.product(range(self.n_node), range(3)):
                quad_point_global[i] += N[I]*self.nodes[I].coor[i]

            quad_point.set_global_coor(quad_point_global)

        self.quad_intp_matrix = np.zeros((self.n_quad_point, self.n_node))
        for i, quad_point in enumerate(self.quad_points):
            for j, shape_function in enumerate(self.N):
                self.quad_intp_matrix[i,j] = shape_function(quad_point.coor)

        self.quad_intp_matrix_inv = inverse(self.quad_intp_matrix)

    def jacobian(self, xi):
        J = np.zeros((self.spatial_dimension,self.elem_dim))

        for I in range(len(self.nodes)):
            for i in range(self.spatial_dimension):
                for j in range(self.elem_dim):
                    J[i,j] += self.nodes[I].coor[i]*self.Bhat[I](xi)[j]

        return J


    def interpolate_at_quad_point(self, function, quad_point_idx):
        result = np.zeros((function.function_size,1))

        for I in range(self.n_node):
            val = function.get_node_val(self.nodes[I].idx)
            # shape_function_val = self.N[I](self.quad_points[quad_point_idx].coor)

            for i in range(function.function_size):
                result[i] += self.quad_points[quad_point_idx].N[I]*val[i]

        return result

    def interpolate_deriv_at_quad_point(self, function, quad_point_idx):
        result = np.zeros((function.function_size,self.spatial_dimension))

        for I in range(self.n_node):
            val = function.get_node_val(self.nodes[I].idx)
            # shape_function_val = self.N[I](self.quad_points[quad_point_idx].coor)
            for i in range(function.function_size):
                for j in range(self.spatial_dimension):
                    result[i,j] += self.quad_points[quad_point_idx].B[I][j]*val[i]

        # import ipdb; ipdb.set_trace()

        return result


