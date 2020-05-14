from lyza import *
from reaction_advection_diffusion import *

# import logging
# logging.getLogger().setLevel(level=logging.DEBUG)

RESOLUTIONS = [4, 6, 8, 10, 15, 20, 30, 40]
# RESOLUTIONS = [4, 6, 8, 10]

n_node_array = []
h_max_array = []

l2_array = []
h1_array = []

for RESOLUTION in RESOLUTIONS:

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    mesh.set_quadrature_degree(lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION)

    a = RADMatrix(mesh, FUNCTION_SIZE)
    m = matrix_assemblers.MassMatrix(mesh, FUNCTION_SIZE)

    b = vector_assemblers.FunctionVector(mesh, FUNCTION_SIZE)
    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    t_array = np.linspace(0, T_MAX, T_RESOLUTION + 1)
    u, f = time_integration.implicit_euler(
        m, a, b, dirichlet_bcs, analytic_solution, t_array
    )

    h_max = 1.0 / RESOLUTION
    n_node = len(mesh.nodes)
    l2 = error.absolute_error(
        u, analytic_solution, analytic_solution_gradient, error="l2", time=T_MAX
    )
    h1 = error.absolute_error(
        u, analytic_solution, analytic_solution_gradient, error="h1", time=T_MAX
    )

    h_max_array.append(h_max)
    n_node_array.append(n_node)
    l2_array.append(l2)
    h1_array.append(h1)

# import ipdb; ipdb.set_trace()
import matplotlib

matplotlib.use("Qt4Agg")
matplotlib.rc("text", usetex=True)

error.plot_errors("plot_errors.pdf", h_max_array, l2=l2_array, h1=h1_array)
error.plot_convergence_rates(
    "plot_convergence_rates.pdf", h_max_array, l2=l2_array, h1=h1_array
)
