from lyza_prototype import *
from math import *
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)



class NonlinearPoissonMatrix(ElementInterface):

    def bilinear_form_matrix(self, elem1, elem2):

        K = np.zeros((elem2.n_dof, elem1.n_dof))

        for q1, q2 in zip(elem1.quad_points, elem2.quad_points):

            for I,J,i in itertools.product(
                    range(elem1.n_node),
                    range(elem2.n_node),
                    range(elem1.physical_dimension)):

                K[I, J] += q1.B[I][i]*q2.B[J][i]*q1.det_jac*q1.weight

        return K


RESOLUTION = 10

exact_solution = lambda x: [sin(2.*pi*x[0])*sin(2.*pi*x[1])]

exact_solution_gradient = lambda x: [[
    2.*pi*cos(2.*pi*x[0])*sin(2.*pi*x[1]),
    2.*pi*sin(2.*pi*x[0])*cos(2.*pi*x[1]),
]]

exact_solution_divgrad = lambda x: 8.*pi*pi*sin(2.*pi*x[0])*sin(2.*pi*x[1])

g = lambda u: sqrt(exp(u))
dgdu = lambda u: 0.5*sqrt(exp(u))

def force_function(x):
    u = exact_solution(x)[0]
    grad_u = exact_solution_gradient(x)[0]
    divgrad_u = exact_solution_divgrad(x)
    grad_g = [dgdu(u) for i in grad_u]

    grad_u_dot_grad_u = sum([i*i for i in grad_u])
    result = dgdu(u)*grad_u_dot_grad_u + g(u)*divgrad_u

    return [result]

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
    a = BilinearForm(V, V, element_matrices.PoissonMatrix(), quadrature_degree)
    b_body_force = LinearForm(V, element_vectors.FunctionElementVector(force_function), quadrature_degree)

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='l2'))
