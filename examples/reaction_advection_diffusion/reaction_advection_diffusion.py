from lyza import *
from lyza.solver import solve_scipy_sparse
from math import *
import itertools
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

QUADRATURE_DEGREE = 1
FUNCTION_SIZE = 1
SPATIAL_DIMENSION = 2

RESOLUTION = 10
# RESOLUTION = 20

PARAM_D = np.eye(2)
# PARAM_D = 1e-6*np.eye(2)
# PARAM_D = np.array([[2., 0.],[0., 1.]])

# PARAM_C = np.array([2., 1.])
PARAM_C = np.array([1., 0.])
# PARAM_C = np.array([0., 0.])

PARAM_R = 1.
# PARAM_R = 0.

T_MAX = 1.
T_RESOLUTION = 50

class RADMatrix(lyza.MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        K = np.zeros((n_dof,n_dof))

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            N = N_arr[idx][:,0]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            K_diffusion = np.einsum('kl,jl,ik->ij', PARAM_D, B, B)*DETJ*W
            K_convection = -1*np.einsum('j,k,ik->ij', N, PARAM_C, B)*DETJ*W
            K_reaction = -1*PARAM_R*np.einsum('j,i->ij', N, N)*DETJ*W

            K += K_diffusion + K_convection + K_reaction

            # for I,J,i,j in itertools.product(
            #         range(self.elements[0].n_node),
            #         range(self.elements[1].n_node),
            #         range(self.elements[0].spatial_dimension),
            #         range(self.elements[0].spatial_dimension)):

            #     K[I, J] += PARAM_D[i,j]*q2.B[J][j]*q1.B[I][i]*q1.det_jac*q1.weight

            # for I,J,i in itertools.product(
            #         range(self.elements[0].n_node),
            #         range(self.elements[1].n_node),
            #         range(self.elements[0].spatial_dimension)):

            #     K[I, J] += -q2.N[J]*PARAM_C[i]*q1.B[I][i]*q1.det_jac*q1.weight

            # for I,J in itertools.product(
            #         range(B.shape[0]),
            #         range(B.shape[0])):
            #     K[I, J] += -PARAM_R*N[J]*N[I]*DETJ*W

        return K

analytic_solution = lambda x, t: [exp(-t)*sin(2.*pi*x[0])*sin(2.*pi*x[1])]

analytic_solution_gradient = lambda x, t: [[
    exp(-t)*2*pi*sin(2*pi*x[1])*cos(2*pi*x[0]),
    exp(-t)*2*pi*sin(2*pi*x[0])*cos(2*pi*x[1]),
]]

force_function = lambda x, t: [
    -(-4*pi**2*PARAM_D[0,0]*sin(2*pi*x[0])*sin(2*pi*x[1])
      + 4*pi**2*PARAM_D[0,1]*cos(2*pi*x[0])*cos(2*pi*x[1])
      + 4*pi**2*PARAM_D[1,0]*cos(2*pi*x[0])*cos(2*pi*x[1])
      - 4*pi**2*PARAM_D[1,1]*sin(2*pi*x[0])*sin(2*pi*x[1])
      - 2*pi*PARAM_C[0]*sin(2*pi*x[1])*cos(2*pi*x[0])
      - 2*pi*PARAM_C[1]*sin(2*pi*x[0])*cos(2*pi*x[1])
      + PARAM_R*sin(2*pi*x[0])*sin(2*pi*x[1])
      + sin(2*pi*x[0])*sin(2*pi*x[1]))*exp(-t)
]

bottom_boundary = lambda x, t: x[1] <= 1e-12
top_boundary = lambda x, t: x[1] >= 1. -1e-12
left_boundary = lambda x, t: x[0] <= 1e-12
right_boundary = lambda x, t: x[0] >= 1.-1e-12
perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])
# perimeter = lambda x: True


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.set_quadrature_degree(lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION)

    a = RADMatrix(mesh, FUNCTION_SIZE)
    m = matrix_assemblers.MassMatrix(mesh, FUNCTION_SIZE)

    b = vector_assemblers.FunctionVector(mesh, FUNCTION_SIZE)
    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    t_array = np.linspace(0, T_MAX, T_RESOLUTION+1)
    u, f = time_integration.implicit_euler(m, a, b, dirichlet_bcs, analytic_solution, t_array)

    ofile = VTKFile('out_rad.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, error='l2', time=T_MAX))
