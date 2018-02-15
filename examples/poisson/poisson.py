from lyza_prototype import *
from math import *
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 10

exact_solution = lambda x: [sin(2.*pi*x[0])*sin(2.*pi*x[1])]

exact_solution_deriv = lambda x: [[
    2.*pi*cos(2.*pi*x[0])*sin(2.*pi*x[1]),
    2.*pi*sin(2.*pi*x[0])*cos(2.*pi*x[1]),
]]

force_function = lambda x: [8.*pi*pi*sin(2.*pi*x[0])*sin(2.*pi*x[1])]

bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12

class PoissonMatrix(lyza_prototype.ElementMatrix):

    def calculate(self, elem1, elem2):
        n_node_1 = len(elem1.nodes)
        n_node_2 = len(elem2.nodes)

        n_dof_1 = n_node_1*elem1.function_dimension
        n_dof_2 = n_node_2*elem2.function_dimension

        K = np.zeros((n_dof_2,n_dof_1))

        for q1, q2 in zip(elem1.quad_points, elem2.quad_points):

            for I,J,i in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(elem1.physical_dimension)):

                K[I, J] += q1.B[I][i]*q2.B[J][i]*q1.det_jac*q1.weight


        # import ipdb; ipdb.set_trace()
        return K


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quadrature_degree = 1
    function_dimension = 1
    physical_dimension = 2
    element_degree = 1

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V)
    a.set_element_matrix(PoissonMatrix(), quadrature_degree)
    b_body_force = LinearForm(V)
    b_body_force.set_element_vector(element_vectors.FunctionElementVector(force_function), quadrature_degree)

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_deriv, quadrature_degree, error='l2'))
