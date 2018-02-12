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


mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)


V = FunctionSpace(mesh, 2, 2, 1, 1)
u = Function(V)
a = BilinearForm(element_matrices.LinearElasticityMatrix(LAMBDA, MU))
b_body_force = LinearForm(element_vectors.FunctionVector(force_function))


bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12

perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])


dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]
# dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

solve(a, b_body_force, u, dirichlet_bcs)
# mesh.write_vtk('out_convergence_test.vtk')

h_max = 1./RESOLUTION
n_node = len(mesh.nodes)
l2 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='l2')
linf = error.absolute_error(u, exact_solution, exact_solution_deriv, error='linf')
h1 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='h1')

print(l2, linf, h1)

