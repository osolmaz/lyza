from lyza_prototype import *
import sympy as sp
import numpy as np

import itertools
import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


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

quadrature_degree = 1
function_size = 1
spatial_dimension = 2
element_degree = 1


class PoissonMatrix(MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.quantity_dict['W'].get_quantity(cell)
        B_arr = self.quantity_dict['B'].get_quantity(cell)
        DETJ_arr = self.quantity_dict['DETJ'].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            for I,J,i in itertools.product(
                    range(n_node),
                    range(n_node),
                    range(B.shape[1])):

                K[I, J] += B[I,i]*B[J,i]*DETJ*W

        return K

class FunctionVector(VectorAssembler):
    def set_param(self, function, time):
        self.function = function
        self.time = time

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.quantity_dict['W'].get_quantity(cell)
        N_arr = self.quantity_dict['N'].get_quantity(cell)
        B_arr = self.quantity_dict['B'].get_quantity(cell)
        DETJ_arr = self.quantity_dict['DETJ'].get_quantity(cell)
        XG_arr = self.quantity_dict['XG'].get_quantity(cell)

        f = np.zeros((n_dof,1))

        for idx in range(len(W_arr)):
            f_val = self.function(XG_arr[idx], self.time)
            N = N_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            for I, i in itertools.product(range(n_node), range(self.function_size)):
                alpha = I*self.function_size + i
                f[alpha] += f_val[i]*N[I,0]*DETJ*W

        # import ipdb; ipdb.set_trace()
        return f


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quantity_dict = mesh.get_basic_quantities(lambda c: quadrature_degree, spatial_dimension)

    a = PoissonMatrix(mesh, function_size, quantity_dict=quantity_dict)
    b = FunctionVector(mesh, function_size, quantity_dict=quantity_dict)

    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b, dirichlet_bcs, solver='petsc')

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    print(max(u.vector), min(u.vector))
    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quantity_dict, error='l2'))

    # print('L2 Error: %e'%error.absolute_error_lp(u, analytic_solution, 2, quantity_dict))
    # print('L2 Error: %e'%error.absolute_error_deriv_lp(u, analytic_solution_gradient, 2, quantity_dict))
