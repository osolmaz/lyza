from math import sqrt, cos, sin, pi
import numpy as np
import itertools


def inverse(J):
    if J.shape[0] == J.shape[1]:
        return np.linalg.inv(J)
    else:
        return np.linalg.inv(J.transpose().dot(J)).dot(J.transpose())

def determinant(J):
    if J.shape[0] == J.shape[1]:
        return np.linalg.det(J)
    else:
        return sqrt(np.linalg.det(J.transpose().dot(J)))


class FiniteElement:
    N = []
    Bhat = []
    quad_weights = []
    quad_points = []
    elem_dim = None

    def __init__(self, nodes, function_space, label=None):
        self.function_space = function_space
        self.quadrature_degree = function_space.quadrature_degree
        self.function_dimension = self.function_space.get_dimension()
        self.physical_dimension = self.function_space.physical_dimension
        self.set_quad_points()

        self.label = label
        self.nodes = nodes


        self.dofmap = []
        for n in self.nodes:
            # node_dofs = [n.idx*self.function_dim+i for i in range(self.function_dim)]
            self.dofmap += self.function_space.node_dofs[n.idx]

        if not self.N or not self.Bhat or not self.quad_weights or not self.quad_points:
            raise Exception('Improper element subclassing')

        # Calculate quadrature point related quantities
        self.quad_jac = []
        self.quad_det_jac = []
        self.quad_jac_inv_tra = []
        self.quad_B = []
        self.quad_N = []
        self.quad_points_global = []

        self.n_quad_point = len(self.quad_points)
        self.n_node = len(self.nodes)


        for quad_point in self.quad_points:
            B = []
            N = []
            jac = self.jacobian(quad_point)
            det_jac = determinant(jac)
            jac_inv_tra = inverse(jac).transpose()

            # import ipdb; ipdb.set_trace()
            self.quad_jac.append(jac)
            self.quad_det_jac.append(det_jac)
            self.quad_jac_inv_tra.append(self.quad_jac_inv_tra)

            for I in range(len(self.N)):
                B.append(jac_inv_tra.dot(self.Bhat[I](quad_point)))
                N.append(self.N[I](quad_point))

            self.quad_B.append(B)
            self.quad_N.append(N)

            quad_point_global = [0. for i in range(self.physical_dimension)]
            for I, i in itertools.product(range(self.n_node), range(self.physical_dimension)):
                quad_point_global[i] += N[I]*self.nodes[I].coor[i]
            self.quad_points_global.append(quad_point_global)

        self.quad_intp_matrix = np.zeros((self.n_quad_point, self.n_node))
        for i, quad_point in enumerate(self.quad_points):
            for j, shape_function in enumerate(self.N):
                self.quad_intp_matrix[i,j] = shape_function(quad_point)

        # try:
        self.quad_intp_matrix_inv = inverse(self.quad_intp_matrix)
        # except:
            # import ipdb; ipdb.set_trace()

    def set_quad_points(self):
        pass

    def jacobian(self, xi):
        J = np.zeros((self.physical_dimension,self.elem_dim))

        for I in range(len(self.nodes)):
            for i in range(self.physical_dimension):
                for j in range(self.elem_dim):
                    J[i,j] += self.nodes[I].coor[i]*self.Bhat[I](xi)[j]

        return J

    def calc_matrix(self, quad_matrix):
        n_dof = len(self.nodes)*self.function_dimension
        K = np.zeros((n_dof,n_dof))
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            K_cont = np.zeros((n_dof,n_dof))
            quad_matrix.eval(
                K_cont,
                self.quad_N[n],
                self.quad_B[n],
                self.quad_det_jac[n],
                self.quad_points_global[n],
                self.function_dimension,
                self.physical_dimension,
                self.elem_dim,
                n_dof,
                n_node)

            K = K + self.quad_weights[n]*K_cont
        # import ipdb; ipdb.set_trace()
        return K

    def calc_vector(self, quad_func):
        n_node = len(self.nodes)
        n_dof = len(self.nodes)*self.function_dimension

        f = np.zeros((n_dof,1))

        for n in range(self.n_quad_point):
            f_cont = np.zeros((n_dof,1))
            quad_func.eval(
                f_cont,
                self.quad_N[n],
                self.quad_B[n],
                self.quad_det_jac[n],
                self.quad_points_global[n],
                self.function_dimension,
                self.physical_dimension,
                self.elem_dim,
                n_dof,
                n_node)

            f = f + self.quad_weights[n]*f_cont

            # if self.elem_dim == 1:
            #     import ipdb; ipdb.set_trace()
            #     print(f_val)


        # import ipdb; ipdb.set_trace()
        return f

    def absolute_error_lp(self, exact, coefficients, p):
        result = 0.
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            u_h = [0. for i in range(self.function_dimension)]

            for I, i in itertools.product(range(n_node), range(self.function_dimension)):
                u_h[i] += self.quad_N[n][I]*coefficients[I*self.function_dimension+i]

            exact_val = exact(self.quad_points_global[n])

            inner_product = 0.
            for i in range(self.function_dimension):
                inner_product += (exact_val[i] - u_h[i])**2

            result += pow(inner_product, p/2.)*self.quad_weights[n]*self.quad_det_jac[n]

        return result

    def absolute_error_deriv_lp(self, exact_deriv, coefficients, p):
        result = 0.
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            u_h = np.zeros((self.function_dimension, self.physical_dimension))

            for I, i, j in itertools.product(range(n_node), range(self.function_dimension), range(self.physical_dimension)):
                u_h[i][j] += self.quad_B[n][I][j]*coefficients[I*self.function_dimension+i]

            exact_val = np.array(exact_deriv(self.quad_points_global[n]))
            # import ipdb; ipdb.set_trace()

            inner_product = 0.
            for i in range(self.function_dimension):
                for j in range(self.physical_dimension):
                    inner_product += (exact_val[i,j] - u_h[i,j])**2

            result += pow(inner_product, p/2.)*self.quad_weights[n]*self.quad_det_jac[n]

        return result


