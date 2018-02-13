from lyza_prototype import *
from math import *
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 5

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

class PoissonMatrix(lyza_prototype.MatrixInterface):

    def calculate(self, elem1, elem2):
        n_node_1 = len(elem1.nodes)
        n_node_2 = len(elem2.nodes)

        n_dof_1 = n_node_1*elem1.function_dimension
        n_dof_2 = n_node_2*elem2.function_dimension

        K = np.zeros((n_dof_2,n_dof_1))

        for n in range(elem1.n_quad_point):
            # K_cont = np.zeros((n_dof,n_dof))


            for I,J,i in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(elem1.physical_dimension)):
                # import ipdb; ipdb.set_trace()

                K[I, J] += elem1.quad_B[n][I][i]*elem2.quad_B[n][J][i]*elem1.quad_det_jac[n]*elem1.quad_points[n].weight



            # quad_matrix.eval(
            #     K_cont,
            #     self.quad_N[n],
            #     self.quad_B[n],
            #     self.quad_det_jac[n],
            #     self.quad_points_global[n],
            #     self.function_dimension,
            #     self.physical_dimension,
            #     self.elem_dim,
            #     n_dof,
            #     n_node)

            # K = K + *K_cont
        # import ipdb; ipdb.set_trace()
        return K


    # def eval(self, K, N_p, B_p, det_jac, quad_point, function_dim, physical_dim, elem_dim, n_dof, n_node):

    #     return K

if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    V = FunctionSpace(mesh, 1, 2, 1)
    u = Function(V)
    a = BilinearForm(V, V, PoissonMatrix(), 1)
    asd = a.assemble()
    import ipdb; ipdb.set_trace()
    # b_body_force = LinearForm(element_vectors.FunctionVector(force_function))

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_deriv, error='l2'))
