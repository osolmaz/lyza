from math import *
from lyza_prototype import *

import logging
logging.basicConfig(level=logging.INFO)

# Exact solution from Di Pietro & Ern 2015
# force_function = lambda x: [
#     2.*pi*pi*sin(pi*x[0])*sin(pi*x[1]),
#     2.*pi*pi*cos(pi*x[0])*cos(pi*x[1]),
# ]

# exact_solution = lambda x: [
#     sin(pi*x[0])*sin(pi*x[1]) + 0.5/LAMBDA*x[0],
#     cos(pi*x[0])*cos(pi*x[1]) + 0.5/LAMBDA*x[1],
# ]

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


# RESOLUTIONS = [2, 4, 6, 8, 10, 15, 20, 30, 40, 60]
# RESOLUTIONS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50]
RESOLUTIONS = [2, 4, 6, 8, 10]
# RESOLUTIONS = [10]

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []


for RESOLUTION in RESOLUTIONS:
    E = 1.
    NU = 0.3

    LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)
    MU = E/2./(1.+NU)

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

    solve(a, b_body_force, u, dirichlet_bcs)


    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='l2')
    linf = error.absolute_error(u, exact_solution, exact_solution_deriv, error='linf')
    h1 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='h1')

    h_max_array.append(h_max)
    n_node_array.append(n_node)
    l2_array.append(l2)
    linf_array.append(linf)
    h1_array.append(h1)


import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rc('text', usetex=True)

error.plot_errors('plot_errors.pdf', h_max_array, l2_array, linf_array, h1_array)
error.plot_convergence_rates('plot_convergence_rates.pdf', h_max_array, l2_array, linf_array, h1_array)
