from lyza_prototype import *
from nonlinear_poisson import *

# import logging
# logging.getLogger().setLevel(level=logging.DEBUG)


# RESOLUTIONS = [4, 6, 8, 10, 15, 20, 30, 40]
RESOLUTIONS = [4, 6, 8, 10, 15]
# RESOLUTIONS = [4, 8, 16]

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []

quadrature_degree = 1
function_dimension = 1
physical_dimension = 2
element_degree = 1


for RESOLUTION in RESOLUTIONS:

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)
    u = Function(V)
    a = NonlinearBilinearForm(V, V, NonlinearPoissonJacobian(), quadrature_degree)
    b_residual = NonlinearForm(V, NonlinearPoissonResidual(), quadrature_degree)
    b_force = LinearForm(V, element_vectors.FunctionElementVector(force_function), quadrature_degree)

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = nonlinear_solve(a, b_residual, b_force, u, dirichlet_bcs)

    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='l2')
    linf = error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='linf')
    h1 = error.absolute_error(u, exact_solution, exact_solution_gradient, quadrature_degree, error='h1')

    h_max_array.append(h_max)
    n_node_array.append(n_node)
    l2_array.append(l2)
    linf_array.append(linf)
    h1_array.append(h1)

# import ipdb; ipdb.set_trace()
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rc('text', usetex=True)

error.plot_errors('plot_errors.pdf', h_max_array, l2_array, linf_array, h1_array)
error.plot_convergence_rates('plot_convergence_rates.pdf', h_max_array, l2_array, linf_array, h1_array)
