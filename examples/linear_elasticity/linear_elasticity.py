from lyza_prototype import *
from math import *
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 10

E = 1000.
NU = 0.3

MU = E/(1.+NU)/2.
LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)

# def exact_solution(pos):
#     x = pos[0]
#     y = pos[1]
#     return[sin(2*pi*x)*cos(2*pi*y), sin(2*pi*y)*cos(2*pi*x)]

# def exact_solution_gradient(pos):
#     x = pos[0]
#     y = pos[1]
#     return [[2*pi*cos(2*pi*x)*cos(2*pi*y), -2*pi*sin(2*pi*x)*sin(2*pi*y)], [-2*pi*sin(2*pi*x)*sin(2*pi*y), 2*pi*cos(2*pi*x)*cos(2*pi*y)]]

# def force_function(pos):
#     x = pos[0]
#     y = pos[1]

#     return [8*pi**2*E*(NU - 1)*sin(2*pi*x)*cos(2*pi*y)/((NU + 1)*(2*NU - 1)), 8*pi**2*E*(NU - 1)*sin(2*pi*y)*cos(2*pi*x)/((NU + 1)*(2*NU - 1))]


def exact_solution(pos):
    x = pos[0]
    y = pos[1]
    return [0, -x*y*(x - 1)*(y - 1)]

def exact_solution_gradient(pos):
    x = pos[0]
    y = pos[1]
    return [[0, 0], [y*(-2*x + 1)*(y - 1), x*(x - 1)*(-2*y + 1)]]

def force_function(pos):
    x = pos[0]
    y = pos[1]

    return [-E*(4*x*y - 2*x - 2*y + 1)/(4*NU**2 + 2*NU - 2), E*(2*x*(NU - 1)*(x - 1) + y*(2*NU - 1)*(y - 1))/((NU + 1)*(2*NU - 1))]


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

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]
    # dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_linear_elasticity.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='l2'))


