from lyza_prototype import *
from linear_elasticity import *

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTIONS = [2, 4, 8, 16, 32, 64]
# RESOLUTIONS = [2, 4, 6, 8, 10, 20]
# RESOLUTIONS = [10]

n_node_array = []
h_max_array = []

l2_array = []
h1_array = []


for RESOLUTION in RESOLUTIONS:
    logging.info('Solving for resolution %d'%RESOLUTION)

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    a = matrix_assemblers.LinearElasticity(mesh, function_size)
    a.set_param_isotropic(LAMBDA, MU, plane_strain=True)

    b = vector_assemblers.FunctionVector(mesh, function_size)
    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b, dirichlet_bcs)

    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, error='l2')
    h1 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, error='h1')

    h_max_array.append(h_max)
    n_node_array.append(n_node)
    l2_array.append(l2)
    h1_array.append(h1)


import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rc('text', usetex=True)

error.plot_errors('plot_errors.pdf', h_max_array, l2=l2_array, h1=h1_array)
error.plot_convergence_rates('plot_convergence_rates.pdf', h_max_array, l2=l2_array, h1=h1_array)
