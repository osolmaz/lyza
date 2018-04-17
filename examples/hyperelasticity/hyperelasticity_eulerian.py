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

BENDING_LOAD = 2.*HORIZONTAL_WIDTH/2.
AXIAL_LOAD = 10.

left_boundary = lambda x: x[0] <= 1e-12

up_left = lambda x: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12
up_right = lambda x: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12
down_left = lambda x: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] < 1e-12
down_right = lambda x: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] < 1e-12

# down_left = lambda x: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] < VERTICAL_WIDTH - 1e-12

if __name__ == '__main__':

    mesh = Cantilever3D(RESOLUTION, LENGTH, HORIZONTAL_WIDTH, VERTICAL_WIDTH)

    spatial_dimension = 3
    function_size = 3
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    a = BilinearForm(V, V, EulerianHyperElasticityTangent(LAMBDA, MU), quadrature_degree)
    b_res = LinearForm(V, EulerianHyperElasticityResidual(LAMBDA, MU), quadrature_degree)

    b_1 = LinearForm(V, linear_interfaces.PointLoad(up_left,[-AXIAL_LOAD,0.,-BENDING_LOAD]), quadrature_degree)
    b_2 = LinearForm(V, linear_interfaces.PointLoad(up_right,[-AXIAL_LOAD,0.,-BENDING_LOAD]), quadrature_degree)
    b_3 = LinearForm(V, linear_interfaces.PointLoad(down_left,[-AXIAL_LOAD,0.,0.]), quadrature_degree)
    b_4 = LinearForm(V, linear_interfaces.PointLoad(down_right,[-AXIAL_LOAD,0.,0.]), quadrature_degree)

    b_load = b_1+b_2+b_3+b_4

    phi0 = V.get_position_function()
    dirichlet_bcs = [DirichletBC(lambda x, t: [x[0],x[1],x[2]], left_boundary)]

    phi, residual = nonlinear_solve(a, b_res, b_load, phi0, dirichlet_bcs, lambda i: i.phi, lambda i: i.F)

    # Calculate displacement from position
    u = Function(V)
    u.vector = phi.vector - phi0.vector

    # Calculate stresses
    a_calc = BilinearForm(V, V, CalculateStrainStress(LAMBDA, MU), quadrature_degree)
    a_calc.project_gradient_to_quadrature_points(phi, lambda i: i.F)
    a_calc.assemble()

    cauchy_stress = a_calc.project_to_nodes(lambda i: i.sigma)
    pk2_stress = a_calc.project_to_nodes(lambda i: i.S)
    gl_strain = a_calc.project_to_nodes(lambda i: i.E)

    # Calculate internal forces
    f = Function(V)
    f.set_vector(-1*b_res.assemble())

    # Set labels
    u.set_label('u')
    f.set_label('f')
    phi.set_label('phi')
    cauchy_stress.set_label('cauchy_stress')
    pk2_stress.set_label('pk2_stress')
    gl_strain.set_label('gl_strain')

    # Output VTK file
    ofile = VTKFile('out_hyperelastic_eulerian.vtk')

    ofile.write(mesh, [u, phi, f, cauchy_stress, pk2_stress, gl_strain])



