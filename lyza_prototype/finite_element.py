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
        self.function_dimension = self.function_space.get_dimension()
        self.physical_dimension = self.function_space.physical_dimension
        self.quad_points = parent_cell.get_quad_points(quadrature_degree)
        self.n_quad_point = len(self.quad_points)

        self.nodes = nodes
        self.n_node = len(self.nodes)
        self.n_dof = self.n_node*self.function_dimension
        self.parent_cell = parent_cell

        self.dofmap = []
        for n in self.parent_cell.nodes:
            node_dofs = [n.idx*self.function_dimension+i for i in range(self.function_dimension)]
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
        J = np.zeros((self.physical_dimension,self.elem_dim))

        for I in range(len(self.nodes)):
            for i in range(self.physical_dimension):
                for j in range(self.elem_dim):
                    J[i,j] += self.nodes[I].coor[i]*self.Bhat[I](xi)[j]

        return J


    def interpolate_scalar(self, nodal_values, position):
        assert len(nodal_values) == self.n_node

        result = 0.
        for I in range(len(self.n_node)):
            result += self.N[I](position)*nodal_values[I]

        return result

