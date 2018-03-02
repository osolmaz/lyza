from lyza_prototype import *
from reaction_advection_diffusion import *

# import logging
# logging.getLogger().setLevel(level=logging.DEBUG)


RESOLUTIONS = [4, 6, 8, 10, 15, 20]
# RESOLUTIONS = [4, 6, 8, 10]

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []

quadrature_degree = 1
function_size = 1
spatial_dimension = 2
element_degree = 1


for RESOLUTION in RESOLUTIONS:

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

    t_array = np.linspace(0, T_MAX, T_RESOLUTION)
    u, f = implicit_euler(m, a, b, u, dirichlet_bcs, analytic_solution, t_array)

    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2', time=T_MAX)
    linf = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='linf', time=T_MAX)
    h1 = error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='h1', time=T_MAX)

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
