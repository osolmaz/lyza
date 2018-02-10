from math import *
import copy
import os

from pylyza.quadmesh import QuadMesh
from pylyza.boundary_condition import DirichletBC, NeumannBC, join_boundaries
from pylyza.linear_elasticity import LinearElasticityMatrix

import logging
logging.basicConfig(level=logging.INFO)


# RESOLUTIONS = [2, 4, 6, 8, 10, 15, 20, 30, 40, 60]
# RESOLUTIONS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50]
RESOLUTIONS = [2, 4, 6, 8, 10]
# RESOLUTIONS = [10]

errors_file = open('out-errors_convergence_test.csv', 'w')
errors_file.write('# resolution L2_error Linf_error H1_error\n')

n_node_array = []
h_max_array = []

linf_array = []
l2_array = []
h1_array = []

linf_convergence_array = [float('nan')]
l2_convergence_array = [float('nan')]
h1_convergence_array = [float('nan')]


for RESOLUTION in RESOLUTIONS:
    E = 1.
    NU = 0.3

    LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)
    MU = E/2./(1.+NU)

    PARAM = {
        'matrix_quadrature_interface': LinearElasticityMatrix({'lambda':LAMBDA, 'mu':MU}),
        'physical_dim': 2,
        'resolution_x': RESOLUTION,
        'resolution_y': RESOLUTION,
        'p0': [0., 0.],
        'p1': [1., 0.],
        'p2': [1., 1.],
        'p3': [0., 1.],
    }

    mesh = QuadMesh(PARAM)

    bottom_boundary = lambda x: x[1] <= 1e-12
    top_boundary = lambda x: x[1] >= 1. -1e-12
    left_boundary = lambda x: x[0] <= 1e-12
    right_boundary = lambda x: x[0] >= 1.-1e-12

    perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

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

    dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]
    # dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

    mesh.solve(dirichlet_bcs, force_function=force_function)
    # mesh.write_vtk('out_convergence_test.vtk')

    h_max = 1./RESOLUTION
    n_node = len(mesh.nodes)
    l2 = mesh.absolute_error(exact_solution, exact_solution_deriv, error='l2')
    linf = mesh.absolute_error(exact_solution, exact_solution_deriv, error='linf')
    h1 = mesh.absolute_error(exact_solution, exact_solution_deriv, error='h1')

    h_max_array.append(h_max)
    n_node_array.append(n_node)
    l2_array.append(l2)
    linf_array.append(linf)
    h1_array.append(h1)

    if len(h_max_array) >= 2:
        denominator = log(h_max_array[-2]-h_max_array[-1])
        l2_convergence_array.append(log(l2_array[-2]-l2_array[-1])/denominator)
        linf_convergence_array.append(log(linf_array[-2]-linf_array[-1])/denominator)
        h1_convergence_array.append(log(h1_array[-2]-h1_array[-1])/denominator)

    errors_file.write('%d %e %e %e %e %e %e\n'%(
        n_node,
        l2,
        l2_convergence_array[-1],
        linf,
        linf_convergence_array[-1],
        h1,
        h1_convergence_array[-1],
    ))
    errors_file.flush()


import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rc('text', usetex=True)

import pylab as pl

# Error figure
pl.loglog(h_max_array, l2_array, '-o', label='$L^2$ Error')
pl.loglog(h_max_array, linf_array, '-o', label='$L^\infty$ Error')
pl.loglog(h_max_array, h1_array, '-o', label='$H^1$ Error')


# pl.minorticks_on()
pl.xlabel('$h_{max}$')
pl.ylabel('$\epsilon_{a}$')
pl.grid(b=True, which='minor', color='gray', linestyle='--')
pl.grid(b=True, which='major', color='gray', linestyle='-')
pl.title('Errors')
pl.legend()

pl.savefig('plot_convergence_test.pdf')

# Convergence rate figure
pl.figure()

pl.semilogx(h_max_array, l2_convergence_array, '-o', label='$L^2$ Convergence rate')
pl.semilogx(h_max_array, linf_convergence_array, '-o', label='$L^\infty$ Convergence rate')
pl.semilogx(h_max_array, h1_convergence_array, '-o', label='$H^1$ Convergence rate')

pl.xlabel('$h_{max}$')
pl.ylabel('$\log(\epsilon_{n-1}-\epsilon_{n})/\log(h_{max,n-1}-h_{max,n})$')
pl.grid(b=True, which='minor', color='gray', linestyle='--')
pl.grid(b=True, which='major', color='gray', linestyle='-')
pl.title('Convergence rates')
pl.legend()

pl.savefig('plot_convergence_test_rates.pdf')

# pl.show()

