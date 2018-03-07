from lyza_prototype import *
from lyza_prototype.solver import solve_scipy_sparse
from math import *
import itertools
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


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

class RADMatrix(ElementInterface):

    def matrix(self):

        K = np.zeros((self.elements[1].n_dof, self.elements[0].n_dof))

        for q1, q2 in zip(self.elements[0].quad_points, self.elements[1].quad_points):

            for I,J,i,j in itertools.product(
                    range(self.elements[0].n_node),
                    range(self.elements[1].n_node),
                    range(self.elements[0].spatial_dimension),
                    range(self.elements[0].spatial_dimension)):

                K[I, J] += PARAM_D[i,j]*q2.B[J][j]*q1.B[I][i]*q1.det_jac*q1.weight

            for I,J,i in itertools.product(
                    range(self.elements[0].n_node),
                    range(self.elements[1].n_node),
                    range(self.elements[0].spatial_dimension)):

                K[I, J] += -q2.N[J]*PARAM_C[i]*q1.B[I][i]*q1.det_jac*q1.weight

            for I,J in itertools.product(
                    range(self.elements[0].n_node),
                    range(self.elements[1].n_node)):
                K[I, J] += -PARAM_R*q2.N[J]*q1.N[I]*q1.det_jac*q1.weight

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

bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12
perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])
# perimeter = lambda x: True


if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    quadrature_degree = 1
    function_size = 1
    spatial_dimension = 2
    element_degree = 1

    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, RADMatrix(), quadrature_degree)
    m = BilinearForm(V, V, bilinear_interfaces.MassMatrix(), quadrature_degree)
    b = LinearForm(V, linear_interfaces.FunctionInterface(force_function), quadrature_degree)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    t_array = np.linspace(0, T_MAX, T_RESOLUTION+1)
    u, f = time_integration.implicit_euler(m, a, b, u, dirichlet_bcs, analytic_solution, t_array)

    ofile = VTKFile('out_rad.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2', time=T_MAX))
