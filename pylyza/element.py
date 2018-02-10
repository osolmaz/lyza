from math import sqrt, cos, sin, pi
import numpy as np
import itertools

quadrature_weights_1d = [1., 1.]
quadrature_points_1d = [[-1./sqrt(3)], [1./sqrt(3)]]

# quadrature_weights_1d = [5./9., 8./9., 5./9.]
# quadrature_points_1d = [[-sqrt(3./5)], [0.], [sqrt(3./5)]]


quadrature_weights_2d = []
quadrature_points_2d = []
for w1, p1 in zip(quadrature_weights_1d, quadrature_points_1d):
    for w2, p2 in zip(quadrature_weights_1d, quadrature_points_1d):
        quadrature_weights_2d.append(w1*w2)
        quadrature_points_2d.append([p1[0],p2[0]])

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


class Element():
    N = []
    Bhat = []
    quad_weights = []
    quad_points = []
    elem_dim = None

    def __init__(self, nodes, param, label=None):
        self.label = label
        self.nodes = nodes
        self.param = param

        self.physical_dim = self.param['physical_dim']

        self.matrix_quadrature_interface = self.param['matrix_quadrature_interface']

        self.dofmap = []
        for n in self.nodes:
            self.dofmap += n.dofmap

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

            quad_point_global = [0. for i in range(self.physical_dim)]
            for I, i in itertools.product(range(self.n_node), range(self.physical_dim)):
                quad_point_global[i] += N[I]*self.nodes[I].coor[i]
            self.quad_points_global.append(quad_point_global)

    def jacobian(self, xi):
        J = np.zeros((self.physical_dim,self.elem_dim))

        for I in range(len(self.nodes)):
            for i in range(self.physical_dim):
                for j in range(self.elem_dim):
                    J[i,j] += self.nodes[I].coor[i]*self.Bhat[I](xi)[j]

        return J

    def calc_stiffness_matrix(self):
        n_dof = len(self.nodes)*self.physical_dim
        K = np.zeros((n_dof,n_dof))
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            K_cont = np.zeros((n_dof,n_dof))
            self.matrix_quadrature_interface.eval(
                K_cont,
                self.quad_N[n],
                self.quad_B[n],
                self.quad_det_jac[n],
                self.physical_dim,
                self.elem_dim,
                n_dof,
                n_node)

            K = K + self.quad_weights[n]*K_cont
        # import ipdb; ipdb.set_trace()
        return K

    def calc_rhs_vector(self, f_func):
        n_node = len(self.nodes)
        n_dof = len(self.nodes)*self.physical_dim

        f = np.zeros((n_dof,1))

        for n in range(self.n_quad_point):

            f_val = f_func(self.quad_points_global[n])

            # if self.elem_dim == 1:
            #     import ipdb; ipdb.set_trace()
            #     print(f_val)

            for I, i in itertools.product(range(n_node), range(self.physical_dim)):
                alpha = I*self.physical_dim + i
                f[alpha] += f_val[i]*self.quad_N[n][I]*self.quad_det_jac[n]*self.quad_weights[n]

        # import ipdb; ipdb.set_trace()
        return f

    def absolute_error_lp(self, exact, coefficients, p):
        result = 0.
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            u_h = [0. for i in range(self.physical_dim)]

            for I, i in itertools.product(range(n_node), range(self.physical_dim)):
                u_h[i] += self.quad_N[n][I]*coefficients[I*self.physical_dim+i]

            exact_val = exact(self.quad_points_global[n])

            inner_product = 0.
            for i in range(self.physical_dim):
                inner_product += (exact_val[i] - u_h[i])**2

            result += pow(inner_product, p/2.)*self.quad_weights[n]*self.quad_det_jac[n]

        return result

    def absolute_error_deriv_lp(self, exact_deriv, coefficients, p):
        result = 0.
        n_node = len(self.nodes)

        for n in range(self.n_quad_point):
            u_h = np.zeros((self.physical_dim, self.physical_dim))

            for I, i, j in itertools.product(range(n_node), range(self.physical_dim), range(self.physical_dim)):
                u_h[i][j] += self.quad_B[n][I][j]*coefficients[I*self.physical_dim+i]

            exact_val = np.array(exact_deriv(self.quad_points_global[n]))
            # import ipdb; ipdb.set_trace()

            inner_product = 0.
            for i in range(self.physical_dim):
                for j in range(self.physical_dim):
                    inner_product += (exact_val[i,j] - u_h[i,j])**2

            result += pow(inner_product, p/2.)*self.quad_weights[n]*self.quad_det_jac[n]

        return result


class QuadElement(Element):
    N = [
        lambda xi: 0.25*(1.-xi[0])*(1.-xi[1]),
        lambda xi: 0.25*(1.+xi[0])*(1.-xi[1]),
        lambda xi: 0.25*(1.+xi[0])*(1.+xi[1]),
        lambda xi: 0.25*(1.-xi[0])*(1.+xi[1]),
    ]

    Bhat = [
        lambda xi: np.array([-0.25*(1.-xi[1]), -0.25*(1.-xi[0])]),
        lambda xi: np.array([+0.25*(1.-xi[1]), -0.25*(1.+xi[0])]),
        lambda xi: np.array([+0.25*(1.+xi[1]), +0.25*(1.+xi[0])]),
        lambda xi: np.array([-0.25*(1.+xi[1]), +0.25*(1.-xi[0])]),
    ]

    quad_weights = quadrature_weights_2d
    quad_points = quadrature_points_2d
    elem_dim = 2

class LineElement(Element):
    N = [
        lambda xi: 0.5*(1.+xi[0]),
        lambda xi: 0.5*(1.-xi[0]),
    ]

    Bhat = [
        lambda xi: np.array([0.5]),
        lambda xi: np.array([-0.5]),
    ]

    quad_weights = quadrature_weights_1d
    quad_points = quadrature_points_1d
    elem_dim = 1
