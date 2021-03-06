from math import *
from lyza import *
import numpy as np
import itertools

import logging

logging.basicConfig(level=logging.INFO)

# The primary purpose of this example is to have a convergence analysis
# with Timoshenko's analytic solution for cantilever beams
# It is still work in progress.

QUADRATURE_DEGREE = 1
FUNCTION_SIZE = 2
SPATIAL_DIMENSION = 2

L = 4.0
C = 1.0
P = 1.0

E = 10000.0
NU = 0.3
I = 1.0 / 12.0 * C * C * C

MU = E / (1.0 + NU) / 2.0
LAMBDA = E * NU / (1.0 + NU) / (1.0 - 2.0 * NU)


def exact_solution(coor, t):
    x = coor[0]
    y = coor[1]

    # q = P/C
    # delta = 5./24.*q*L*L*L*L/E/I*(1. + 12./5.*C*C/L/L*(4./5. + NU/2.))

    # u = q/2./E/I*((L*L*x - x*x*x/3.)*y + x*(2./3.*y*y*y - 2./5.*C*C*y) \
    #               + NU*x*(1./3.*y*y*y - C*C*y + 2./3.*C*C*C))

    # v = -q/2./E/I*(y*y*y*y/12. - C*C*y*y/2. + 2./3.*C*C*C*y \
    #                + NU*((L*L - x*x)*y*y/2. + y*y*y*y/6. - 1./5.*C*C*y*y)) \
    #                -1/2./E/I*(L*L*x*x/2. - x*x*x*x/12. - 1./5.*C*C*x*x\
    #                           +(1. + 1./2.*NU)*C*C*x*x)\
    #                + delta

    # u = -P*x*x*y/(2.*E*I) - NU*P*y*y*y/(6.*E*I) + P*y*y*y/(6.*I*MU) \
    #     + (P*L*L/(2.*E*I) - P*C*C/(2.*I*MU))*y
    # v = NU*P*x*y*y/(2.*E*I) + P*x*x*x/(6.*E*I) - P*L*L*x/(2.*E*I) + P*L*L*L/(3.*E*I)

    u = (
        -P
        * y
        / 6.0
        / E
        / I
        * ((6.0 * L - 3.0 * x) * x + (2.0 + NU) * (y * y - C * C / 4.0))
    )
    v = (
        -P
        / 6.0
        / E
        / I
        * (
            3.0 * NU * y * y * (L - x)
            + (4.0 + 5.0 * NU) * C * C * x / 4.0
            + (3.0 * L - x) * x * x
        )
    )

    return [u, v]


ZERO_FUNCTION = lambda x, t: [0.0, 0.0, 0.0]
FORCE_FUNCTION = lambda x, t: [0.0, -6.0 * P / C / C / C * (C * C / 4.0 - x[1] * x[1])]
# FORCE_FUNCTION = lambda x, t: [0.,-P/C]


class RightEnd(Domain):
    def is_subset(self, cell):
        is_in = not (False in [right_boundary(node.coor, 0) for node in cell.nodes])

        return is_in and cell.is_boundary


right_boundary = lambda x, t: x[0] >= L - 1e-12
left_boundary = lambda x, t: x[0] <= 1e-12

left_bottom_point = lambda x, t: x[0] <= 1e-12 and x[1] <= -C / 2.0 + 1e-12

mesh = meshes.QuadMesh(
    40, 10, [0.0, -C / 2.0], [L, -C / 2.0], [L, C / 2.0], [0.0, C / 2.0],
)

mesh.set_quadrature_degree(
    lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION, domain=domain.AllDomain()
)

a = matrix_assemblers.LinearElasticityMatrix(mesh, FUNCTION_SIZE)
a.set_param_isotropic(LAMBDA, MU, plane_stress=True)

b_neumann = vector_assemblers.FunctionVector(mesh, FUNCTION_SIZE, domain=RightEnd())
b_neumann.set_param(FORCE_FUNCTION, 0)

# dirichlet_bcs = [DirichletBC(lambda x: [0.,0.], right_boundary)]
dirichlet_bcs = [DirichletBC(ZERO_FUNCTION, left_boundary)]
# dirichlet_bcs = [DirichletBC(exact_solution, left_boundary)]
# dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

u, f = solve(a, b_neumann, dirichlet_bcs)

ofile = VTKFile("out_beam.vtk")

u.set_label("u")
f.set_label("f")

ofile.write(mesh, [u, f])
