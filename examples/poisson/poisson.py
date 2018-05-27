from lyza_prototype import *
import sympy as sp
import numpy as np

import itertools
import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

QUADRATURE_DEGREE = 1
FUNCTION_SIZE = 1
SPATIAL_DIMENSION = 2

# RESOLUTION = 100
RESOLUTION = 10

class PoissonAnalyticSolution(AnalyticSolution):
    def get_force_expression(self):
        f = sp.Matrix([0])

        for i in range(2):
            f[0] += -sp.diff(sp.diff(self.u[0], self.position[i]), self.position[i])

        return f

analytic_sol_expr = lambda x: [sp.sin(2*sp.pi*x[0])*sp.sin(2*sp.pi*x[1])]
# analytic_sol_expr = lambda x: [sp.sin(2*sp.pi*x[0])*sp.cos(2*sp.pi*x[1])]

analytic_solution_obj = PoissonAnalyticSolution(analytic_sol_expr, 1, 2)

analytic_solution = analytic_solution_obj.get_analytic_solution_function()
analytic_solution_gradient = analytic_solution_obj.get_gradient_function()
force_function = analytic_solution_obj.get_rhs_function()

bottom_boundary = lambda x, t: x[1] <= 1e-12
top_boundary = lambda x, t: x[1] >= 1. -1e-12
left_boundary = lambda x, t: x[0] <= 1e-12
right_boundary = lambda x, t: x[0] >= 1.-1e-12
perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])
# perimeter = lambda x: True


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    mesh.set_quadrature_degree(lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION)

    a = matrix_assemblers.PoissonMatrix(mesh, FUNCTION_SIZE)
    b = vector_assemblers.FunctionVector(mesh, FUNCTION_SIZE)

    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b, dirichlet_bcs, solver='petsc')

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, error='l2'))

