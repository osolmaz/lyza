from lyza_prototype import *
import sympy as sp
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)


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


class PoissonMatrix(MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        quad_weight_arr = self.quad_weight.get_quantity(cell)
        B_arr = self.B.get_quantity(cell)
        det_jac_arr = self.det_jac.get_quantity(cell)

        for idx in range(len(quad_weight_arr)):
            B = B_arr[idx]
            w = quad_weight_arr[idx][0,0]
            j = det_jac_arr[idx][0,0]

            for I,J,i in itertools.product(
                    range(n_node),
                    range(n_node),
                    range(B.shape[0])):

                K[I, J] += B[i,I]*B[i,J]*w*j

        # import ipdb; ipdb.set_trace()
        return K

class FunctionVector(VectorAssembler):
    def set_param(self, function, time):
        self.function = function
        self.time = time

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        quad_weight_arr = self.quad_weight.get_quantity(cell)
        N_arr = self.N.get_quantity(cell)
        B_arr = self.B.get_quantity(cell)
        det_jac_arr = self.det_jac.get_quantity(cell)
        global_coor_arr = self.global_coor.get_quantity(cell)

        f = np.zeros((n_dof,1))

        for idx in range(len(quad_weight_arr)):
            f_val = self.function(global_coor_arr[idx], self.time)
            N = N_arr[idx]
            w = quad_weight_arr[idx][0,0]
            j = det_jac_arr[idx][0,0]

            for I, i in itertools.product(range(n_node), range(self.function_size)):
                alpha = I*self.function_size + i
                f[alpha] += f_val[i]*N[0,I]*j*w

        # import ipdb; ipdb.set_trace()
        return f


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quadrature_degree = 1
    function_size = 1
    spatial_dimension = 2
    element_degree = 1

    quad_weight, quad_coor = mesh.get_quadrature_quantities(lambda c: quadrature_degree)

    N, B, jac, det_jac, jac_inv_tra, global_coor = mesh.get_basis_quantities(lambda c: quadrature_degree, spatial_dimension)

    a = PoissonMatrix(mesh, function_size)
    a.set_basic_quantities(N, B, jac, det_jac, jac_inv_tra, global_coor, quad_coor, quad_weight)

    b = FunctionVector(mesh, function_size)
    b.set_basic_quantities(N, B, jac, det_jac, jac_inv_tra, global_coor, quad_coor, quad_weight)
    b.set_param(force_function, 0)

    # import ipdb; ipdb.set_trace()

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    # print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2'))
