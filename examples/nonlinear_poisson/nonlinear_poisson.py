from lyza_prototype import *
from math import *
import numpy as np

import itertools
import logging
logging.basicConfig(level=logging.INFO)

quadrature_degree = 1
function_size = 1
spatial_dimension = 2
element_degree = 1

# exact_solution = lambda x, t: [sin(2.*pi*x[0])*sin(2.*pi*x[1])]
# exact_solution_gradient = lambda x, t: [[
#     2.*pi*cos(2.*pi*x[0])*sin(2.*pi*x[1]),
#     2.*pi*sin(2.*pi*x[0])*cos(2.*pi*x[1]),
# ]]
# exact_solution_divgrad = lambda x, t: -8.*pi*pi*sin(2.*pi*x[0])*sin(2.*pi*x[1])

exact_solution = lambda x, t: [sin(2.*pi*x[0])*cos(2.*pi*x[1])]
exact_solution_gradient = lambda x, t: [[
    2.*pi*cos(2.*pi*x[0])*cos(2.*pi*x[1]),
    -2.*pi*sin(2.*pi*x[0])*sin(2.*pi*x[1]),
]]
exact_solution_divgrad = lambda x, t: -8.*pi*pi*sin(2.*pi*x[0])*cos(2.*pi*x[1])

g_function = lambda u: sqrt(exp(u))
dgdu_function = lambda u: 0.5*sqrt(exp(u))

# g_function = lambda u: 1.
# dgdu_function = lambda u: 0.


def force_function(x, t):
    u = exact_solution(x, t)[0]
    grad_u = exact_solution_gradient(x, t)[0]
    divgrad_u = exact_solution_divgrad(x, t)

    grad_u_dot_grad_u = sum([i*i for i in grad_u])
    result = -(dgdu_function(u)*grad_u_dot_grad_u + g_function(u)*divgrad_u)

    return [result]


class NonlinearPoissonJacobian(MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        U_arr = self.mesh.quantities['U'].get_quantity(cell)
        GRADU_arr = self.mesh.quantities['GRADU'].get_quantity(cell)

        G_arr = self.mesh.quantities['G'].get_quantity(cell)
        DGDU_arr = self.mesh.quantities['DGDU'].get_quantity(cell)

        for idx in range(len(W_arr)):
            N = N_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            u_n = U_arr[idx][0,0]
            grad_u_n = GRADU_arr[idx][0,:]
            g_u_n = G_arr[idx][0,0]
            dgdu_u_n = DGDU_arr[idx][0,0]

            for I,J,i in itertools.product(
                    range(B.shape[0]), range(B.shape[0]), range(B.shape[1])):

                K[I, J] += (dgdu_u_n*N[J,0]*grad_u_n[i]
                            + g_u_n*B[J,i])*B[I,i]*DETJ*W

        return K


class NonlinearPoissonResidual(VectorAssembler):

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        f = np.zeros((n_dof, 1))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        U_arr = self.mesh.quantities['U'].get_quantity(cell)
        GRADU_arr = self.mesh.quantities['GRADU'].get_quantity(cell)

        G_arr = self.mesh.quantities['G'].get_quantity(cell)
        DGDU_arr = self.mesh.quantities['DGDU'].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            u_n = U_arr[idx][0,0]
            grad_u_n = GRADU_arr[idx][0,:]
            g_u_n = G_arr[idx][0,0]
            dgdu_u_n = DGDU_arr[idx][0,0]

            for I,i in itertools.product(
                    range(B.shape[0]),
                    range(B.shape[1])):
                f[I] += -1*g_u_n*grad_u_n[i]*B[I,i]*DETJ*W

        return f

class Calculator(CellIterator):
    def init_quantities(self):
        self.mesh.quantities['G'] = CellQuantity(self.mesh, (1, 1))
        self.mesh.quantities['DGDU'] = CellQuantity(self.mesh, (1, 1))

    def iterate(self, cell):

        U_arr = self.mesh.quantities['U'].get_quantity(cell)

        self.mesh.quantities['G'].reset_quantity_by_cell(cell)
        self.mesh.quantities['DGDU'].reset_quantity_by_cell(cell)

        for U in U_arr:
            u = U[0,0]
            g = np.array([[g_function(u)]])
            dgdu = np.array([[dgdu_function(u)]])
            self.mesh.quantities['G'].add_quantity_by_cell(cell, g)
            self.mesh.quantities['DGDU'].add_quantity_by_cell(cell, dgdu)



RESOLUTION = 10

bottom_boundary = lambda x, t: x[1] <= 1e-12
top_boundary = lambda x, t: x[1] >= 1. -1e-12
left_boundary = lambda x, t: x[0] <= 1e-12
right_boundary = lambda x, t: x[0] >= 1.-1e-12


def update_function(mesh, u):
    projector = iterators.Projector(mesh, u.function_size)
    projector.set_param(u, 'U')
    projector.execute()

    projector = iterators.GradientProjector(mesh, u.function_size)
    projector.set_param(u, 'GRADU', spatial_dimension)
    projector.execute()

    calculator = Calculator(mesh, u.function_size)
    calculator.init_quantities()
    calculator.execute()


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    a = NonlinearPoissonJacobian(mesh, function_size)
    b_residual = NonlinearPoissonResidual(mesh, function_size)
    b_force = vector_assemblers.FunctionVector(mesh, function_size)
    b_force.set_param(force_function, 0)

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = nonlinear_solve(a, b_residual, b_force, dirichlet_bcs, update_function=update_function)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_gradient, error='l2'))
