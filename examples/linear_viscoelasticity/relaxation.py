from lyza_prototype import *
from lyza_prototype.solver import solve_scipy_sparse
from math import *
import itertools
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

# RESOLUTION = 10
RESOLUTION = 20

T_MAX = 50.
DT = 0.5

E = 1000.
NU = 0.45
ETA = 10000.

MU = elasticity.mu_from_E_nu(E, NU)
LAMBDA = elasticity.lambda_from_E_nu(E, NU)

CREEP_DISTANCE = 1.
CREEP_TIME = 10.

# def right_bc_function(x, t):
#     if t < CREEP_TIME:
#         return [CREEP_DISTANCE*t/CREEP_TIME, 0]
#     else:
#         return [CREEP_DISTANCE, 0]

def right_bc_function(x, t):
    return [CREEP_DISTANCE, 0]

INITIAL_CONDITION = lambda x, t: [0., 0.]


class MassMatrix(bilinear_interfaces.LinearElasticity):
    def __init__(self, eta, plane_stress=False, plane_strain=False):
        # matrix = np.array([
        #     [2./3., -1./3., -1./3., 0., 0., 0.],
        #     [-1./3., 2./3., -1./3., 0., 0., 0.],
        #     [-1./3., -1./3., 2./3., 0., 0., 0.],
        #     [0., 0., 0., 1./2., 0., 0.],
        #     [0., 0., 0., 0., 1./2., 0.],
        #     [0., 0., 0., 0., 0., 1./2.],
        # ])
        matrix = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1./2., 0., 0.],
            [0., 0., 0., 0., 1./2., 0.],
            [0., 0., 0., 0., 0., 1./2.],
        ])
        matrix = eta*matrix
        super().__init__(matrix, plane_stress=plane_stress, plane_strain=plane_strain)


bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12

quadrature_degree = 1
function_size = 2
spatial_dimension = 2
element_degree = 1

if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)


    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, bilinear_interfaces.IsotropicLinearElasticity(LAMBDA, MU, plane_stress=True), quadrature_degree)
    m = BilinearForm(V, V, MassMatrix(ETA, plane_stress=True), quadrature_degree)
    b = LinearForm(V, linear_interfaces.ZeroVector(), quadrature_degree)

    dirichlet_bcs = [
        DirichletBC(lambda x,t: [0.,0.], left_boundary),
        DirichletBC(right_bc_function, right_boundary),
    ]

    t_array = time_integration.time_array(0., T_MAX, DT)
    u, f = time_integration.implicit_euler(
        m, a, b, u, dirichlet_bcs,
        INITIAL_CONDITION, t_array, out_prefix='out_relaxation')

    u.set_label('u')


