from lyza_prototype import *
from math import *
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)

exact_solution = lambda x: [sin(2.*pi*x[0])*sin(2.*pi*x[1])]

exact_solution_gradient = lambda x: [[
    2.*pi*cos(2.*pi*x[0])*sin(2.*pi*x[1]),
    2.*pi*sin(2.*pi*x[0])*cos(2.*pi*x[1]),
]]

exact_solution_divgrad = lambda x: 8.*pi*pi*sin(2.*pi*x[0])*sin(2.*pi*x[1])

g = lambda u: sqrt(exp(u))
dgdu = lambda u: 0.5*sqrt(exp(u))

# g = lambda u: 1.
# dgdu = lambda u: 0.


def force_function(x):
    u = exact_solution(x)[0]
    grad_u = exact_solution_gradient(x)[0]
    divgrad_u = exact_solution_divgrad(x)
    grad_g = [dgdu(u) for i in grad_u]

    grad_u_dot_grad_u = sum([i*i for i in grad_u])
    result = dgdu(u)*grad_u_dot_grad_u + g(u)*divgrad_u

    return [result]


class NonlinearBilinearForm(BilinearForm):
    def set_prev_sol(self, function):
        for i in self.interfaces:
            i.prev_sol.vectors = []
            i.prev_sol_deriv.vectors = []
            for quad_point_idx in range(i.n_quad_point):
                i.prev_sol.vectors.append(i.elem1.interpolate_at_quad_point(function, quad_point_idx))
                i.prev_sol_deriv.vectors.append(i.elem1.interpolate_deriv_at_quad_point(function, quad_point_idx))
            # import ipdb; ipdb.set_trace()

            # print(quad_point_idx)

class NonlinearForm(LinearForm):
    def set_prev_sol(self, function):
        for i in self.interfaces:
            i.prev_sol.vectors = []
            i.prev_sol_deriv.vectors = []

            for quad_point_idx in range(i.n_quad_point):
                i.prev_sol.vectors.append(i.elem.interpolate_at_quad_point(function, quad_point_idx))
                i.prev_sol_deriv.vectors.append(i.elem.interpolate_deriv_at_quad_point(function, quad_point_idx))


class NonlinearPoissonJacobian(BilinearElementInterface):

    def init_quadrature_point_quantities(self, n_quad_point):
        self.prev_sol = Quantity((1, 1), n_quad_point)
        self.prev_sol_deriv = Quantity((1, 2), n_quad_point)

    def matrix(self):
        K = np.zeros((self.elem2.n_dof, self.elem1.n_dof))

        for n, (q1, q2) in enumerate(zip(self.elem1.quad_points, self.elem2.quad_points)):

            u_n = self.prev_sol.vectors[n][0,0]
            grad_u_n = self.prev_sol_deriv.vectors[n][0,:]
            g_u_n = g(u_n)
            dgdu_u_n = dgdu(u_n)

            for I,J,i in itertools.product(
                    range(self.elem1.n_node),
                    range(self.elem2.n_node),
                    range(self.elem1.physical_dimension)):

                # K[I, J] += q1.B[I][i]*q2.B[J][i]*q1.det_jac*q1.weight
                K[I, J] += (dgdu_u_n*q1.N[J]*grad_u_n[i]
                            + g_u_n*q1.B[J][i])*q2.B[I][i] * q1.det_jac*q1.weight

        # if self.elem1.parent_cell.idx == 0:
        #     import ipdb; ipdb.set_trace()

        return K


class NonlinearPoissonResidual(LinearElementInterface):

    def init_quadrature_point_quantities(self, n_quad_point):
        self.prev_sol = Quantity((2, 1), n_quad_point)
        self.prev_sol_deriv = Quantity((1, 2), n_quad_point)

    def vector(self):

        f = np.zeros((self.elem.n_dof, 1))

        for n, q in enumerate(self.elem.quad_points):
            u_n = self.prev_sol.vectors[n][0,0]
            grad_u_n = self.prev_sol_deriv.vectors[n][0,:]
            g_u_n = g(u_n)
            # dgdu_u_n = dgdu(u_n)

            for I,i in itertools.product(
                    range(self.elem.n_node),
                    range(self.elem.physical_dimension)):
                f[I] += -1*g_u_n*grad_u_n[i]*q.B[I][i] * q.det_jac*q.weight

        return f


RESOLUTION = 10


bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quadrature_degree = 1
    function_dimension = 1
    physical_dimension = 2
    element_degree = 1

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = NonlinearBilinearForm(V, V, NonlinearPoissonJacobian(), quadrature_degree)
    b_residual = NonlinearForm(V, NonlinearPoissonResidual(), quadrature_degree)
    b_force = LinearForm(V, element_vectors.FunctionElementVector(force_function), quadrature_degree)

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = nonlinear_solve(a, b_residual, b_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='l2'))
