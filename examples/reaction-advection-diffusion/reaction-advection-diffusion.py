from lyza_prototype import *
from lyza_prototype.solver import solve_scipy_sparse
from math import *
import itertools
# import sympy as sp
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 20

PARAM_D = np.eye(2)
PARAM_C = np.array([0., 0.])
# PARAM_C = np.array([1., 0.])
PARAM_R = 1.

T_GLOBAL = 0.

T_MAX = 1.
T_RESOLUTION = 100.

DELTA_T = T_MAX/T_RESOLUTION

# class RADAnalyticSolution(AnalyticSolution):
#     def get_force_expression(self):
#         f = sp.Matrix([0])

#         for i in range(2):
#             f[0] += -sp.diff(sp.diff(self.u[0], self.position[i]), self.position[i])

#         return f

# analytic_sol_expr = lambda x: [sp.sin(2*sp.pi*x[0])*sp.sin(2*sp.pi*x[1])]

# analytic_solution_obj = PoissonAnalyticSolution(analytic_sol_expr, 1, 2)

class RADMatrix(BilinearElementInterface):

    def matrix(self):

        K = np.zeros((self.elem2.n_dof, self.elem1.n_dof))

        for q1, q2 in zip(self.elem1.quad_points, self.elem2.quad_points):

            for I,J,i,j in itertools.product(
                    range(self.elem1.n_node),
                    range(self.elem2.n_node),
                    range(self.elem1.spatial_dimension),
                    range(self.elem1.spatial_dimension)):

                K[I, J] += (PARAM_D[i,j]*q2.B[J][j] - q2.N[J]*PARAM_C[i])*q1.B[I][i]*q1.det_jac*q1.weight

            for I,J in itertools.product(
                    range(self.elem1.n_node),
                    range(self.elem2.n_node)):
                K[I, J] += PARAM_R*q2.N[J]*q1.N[I]*q1.det_jac*q1.weight

        return K


analytic_solution = lambda x: [exp(-T_GLOBAL)*sin(2.*pi*x[0])*sin(2.*pi*x[1])]
analytic_solution_gradient = lambda x: [[exp(-T_GLOBAL)*2*pi*sin(2*pi*x[1])*cos(2*pi*x[0]), 2*pi*sin(2*pi*x[0])*cos(2*pi*x[1])]]
# force_function = lambda x: [exp(-T_GLOBAL)*8*pi**2*sin(2*pi*x[0])*sin(2*pi*x[1])]

# analytic_sol_expr =


bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12
perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])
# perimeter = lambda x: True


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quadrature_degree = 1
    function_size = 1
    spatial_dimension = 2
    element_degree = 1

    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, RADMatrix(), quadrature_degree)
    m = BilinearForm(V, V, bilinear_interfaces.MassMatrix(), quadrature_degree)
    b_body_force = LinearForm(V, linear_interfaces.FunctionElementVector(force_function), quadrature_degree)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    A = a.assemble()
    M = m.assemble()

    u0 = get_analytic_solution_vector(V, analytic_solution)

    t_array = np.linspace(0, T_MAX, T_RESOLUTION)

    solution_vector = u0
    for t in t_array:
        T_GLOBAL = t
        b = b_body_force.assemble()
        matrix = M + DELTA_T*A
        vector = M.dot(u0) + DELTA_T*b

        matrix_bc, vector_bc = apply_bcs(matrix, vector, V, dirichlet_bcs)
        solution_vector = solve_scipy_sparse(matrix_bc, vector_bc)

        logging.info('T = %f'%(T_GLOBAL))

    u.set_vector(solution_vector)
    f = Function(V)
    # f.set_vector()
    # import ipdb; ipdb.set_trace()

    # u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_rad.vtk')

    u.set_label('u')
    # f.set_label('f')

    ofile.write(mesh, u)
    # ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2'))
