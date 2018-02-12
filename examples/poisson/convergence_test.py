from lyza_prototype import *
from poisson import *

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTIONS = [4, 6, 8, 10, 15, 20, 30, 40]
# RESOLUTIONS = [4, 6, 8, 10]
# RESOLUTIONS = [4, 8, 16]

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []

linf_convergence_array = [float('nan')]
l2_convergence_array = [float('nan')]
h1_convergence_array = [float('nan')]


for RESOLUTION in RESOLUTIONS:

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    V = FunctionSpace(mesh, 1, 2, 1, 1)
    u = Function(V)
    a = BilinearForm(PoissonMatrix())
    b_body_force = LinearForm(element_vectors.FunctionVector(force_function))

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_poisson.vtk')

    u.set_label('u')
    f.set_label('f')

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
