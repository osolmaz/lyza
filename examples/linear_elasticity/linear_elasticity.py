from lyza_prototype import *
import numpy as np
from analytic_solution import get_analytic_solution_function, get_gradient_function, get_force_function
from sympy import Symbol, sin, cos, pi
# from math import *

import logging
logging.basicConfig(level=logging.INFO)

RESOLUTION = 10

E = 1000.
NU = 0.3

MU = E/(1.+NU)/2.
LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)

x = Symbol('x')
y = Symbol('y')
# analytic_sol_expr = [sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*sin(2*pi*y)]
# analytic_sol_expr = [sin(2*pi*x)*cos(2*pi*y), sin(2*pi*y)*cos(2*pi*x)]
analytic_sol_expr = [0, -x*y*(x - 1)*(y - 1)]

analytic_solution = get_analytic_solution_function(analytic_sol_expr)
analytic_solution_gradient = get_gradient_function(analytic_sol_expr)
force_function = get_force_function(analytic_sol_expr, E, NU, plane_stress=False)


if __name__ == '__main__':

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    physical_dimension = 2
    function_dimension = 2
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, bilinear_interfaces.IsotropicLinearElasticity(LAMBDA, MU, plane_strain=True), quadrature_degree)
    b_body_force = LinearForm(V, linear_interfaces.FunctionElementVector(force_function), quadrature_degree)

    bottom_boundary = lambda x: x[1] <= 1e-12
    top_boundary = lambda x: x[1] >= 1. -1e-12
    left_boundary = lambda x: x[0] <= 1e-12
    right_boundary = lambda x: x[0] >= 1.-1e-12

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]
    # dirichlet_bcs = [DirichletBC(analytic_solution, lambda x: True)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_linear_elasticity.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2'))


