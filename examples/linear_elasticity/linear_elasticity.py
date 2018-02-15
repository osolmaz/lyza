from lyza_prototype import *

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 10

E = 1.
NU = 0.3

LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)
MU = E/2./(1.+NU)

force_function = lambda x: [
    (LAMBDA+MU)*(1.-2.*x[0])*(1.-2.*x[1]),
    -2.*MU*x[1]*(1.-x[1])-2.*(LAMBDA+2.*MU)*x[0]*(1.-x[0]),
]

exact_solution = lambda x: [
    0.,
    -x[0]*(1-x[0])*x[1]*(1-x[1]),
]

exact_solution_deriv = lambda x: [
    [0.,0.],
    [x[1]*(-2.*x[0]*(x[1]-1.)+x[1]-1.),
     x[0]*(-2.*x[0]*x[1]+x[0]+2.*x[1]-1.)]
]


# Exact solution from Di Pietro & Ern 2015
# force_function = lambda x: [
#     2.*pi*pi*sin(pi*x[0])*sin(pi*x[1]),
#     2.*pi*pi*cos(pi*x[0])*cos(pi*x[1]),
# ]

# exact_solution = lambda x: [
#     sin(pi*x[0])*sin(pi*x[1]) + 0.5/LAMBDA*x[0],
#     cos(pi*x[0])*cos(pi*x[1]) + 0.5/LAMBDA*x[1],
# ]


if __name__ == '__main__':

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    physical_dimension = 2
    function_dimension = 2
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V)
    a.set_element_interface(element_matrices.LinearElasticityMatrix(LAMBDA, MU), quadrature_degree)
    b_body_force = LinearForm(V)
    b_body_force.set_element_interface(element_vectors.FunctionElementVector(force_function), quadrature_degree)

    bottom_boundary = lambda x: x[1] <= 1e-12
    top_boundary = lambda x: x[1] >= 1. -1e-12
    left_boundary = lambda x: x[0] <= 1e-12
    right_boundary = lambda x: x[0] >= 1.-1e-12

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]
    # dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)


    ofile = VTKFile('out_linear_elasticity.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, exact_solution, exact_solution_deriv, quadrature_degree, error='l2'))


