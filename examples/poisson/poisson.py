from lyza_prototype import *
from math import *

import itertools
import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 20

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

    def eval(self, K, N_p, B_p, det_jac, quad_point, function_dim, physical_dim, elem_dim, n_dof, n_node):
        for I,J,i in itertools.product(
                range(n_node),
                range(n_node),
                range(physical_dim)):
            # import ipdb; ipdb.set_trace()

            K[I, J] += B_p[I][i]*B_p[J][i]*det_jac

        return K

if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    V = FunctionSpace(mesh, 1, 2, 1, 1)
    u = Function(V)
    a = BilinearForm(PoissonMatrix())
    b_body_force = LinearForm(element_vectors.FunctionVector(force_function))

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])


