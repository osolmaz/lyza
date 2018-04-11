from lyza_prototype import *
from mesh import Cantilever3D
from interfaces import *

import logging
logging.basicConfig(level=logging.DEBUG)

RESOLUTION = 15
LENGTH = 300.
HORIZONTAL_WIDTH = 20.
VERTICAL_WIDTH = 20.

E = 200.
NU = 0.3

LAMBDA = elasticity.lambda_from_E_nu(E, NU)
MU = elasticity.mu_from_E_nu(E, NU)

LOAD = 2.*HORIZONTAL_WIDTH/2.

left_boundary = lambda x: x[0] <= 1e-12

load_position_left = lambda x: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] < VERTICAL_WIDTH - 1e-12
load_position_right = lambda x: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] < VERTICAL_WIDTH - 1e-12

if __name__ == '__main__':

    mesh = Cantilever3D(RESOLUTION, LENGTH, HORIZONTAL_WIDTH, VERTICAL_WIDTH)

    spatial_dimension = 3
    function_size = 3
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    a = BilinearForm(V, V, EulerianHyperElasticityTangent(LAMBDA, MU), quadrature_degree)
    b_res = LinearForm(V, EulerianHyperElasticityResidual(LAMBDA, MU), quadrature_degree)
    b_1 = LinearForm(V, linear_interfaces.PointLoad(load_position_left,[0.,0.,-LOAD]), quadrature_degree)
    b_2 = LinearForm(V, linear_interfaces.PointLoad(load_position_right,[0.,0.,-LOAD]), quadrature_degree)

    phi0 = V.get_position_function()
    dirichlet_bcs = [DirichletBC(lambda x, t: [x[0],x[1],x[2]], left_boundary)]

    phi, residual = nonlinear_solve(a, b_res, b_1+b_2, phi0, dirichlet_bcs, lambda i: i.phi, lambda i: i.F)

    u = Function(V)
    u.vector = phi.vector - phi0.vector

    ofile = VTKFile('out_hyperelastic_eulerian.vtk')

    u.set_label('u')
    phi.set_label('phi')

    ofile.write(mesh, [u, phi])



