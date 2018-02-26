from lyza_prototype import *
from linear_elasticity import *

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTIONS = [2, 4, 8, 16, 32, 64]
# RESOLUTIONS = [2, 4, 6, 8, 10, 20]
# RESOLUTIONS = [10]

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []


for RESOLUTION in RESOLUTIONS:

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    physical_dimension = 2
    function_dimension = 2
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, bilinear_interfaces.IsotropicLinearElasticity(LAMBDA, MU, plane_strain=True), quadrature_degree)
    b_body_force = LinearForm(V, linear_interfaces.FunctionElementVector(force_function), quadrature_degree)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2')
    linf = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='linf')
    h1 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='h1')

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
