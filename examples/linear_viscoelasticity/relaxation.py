from lyza_prototype import *
from lyza_prototype.solver import solve_scipy_sparse
from math import *
import itertools
import numpy as np
import os

import logging
logging.basicConfig(level=logging.INFO)

# RESOLUTION = 10
RESOLUTION = 20

OUTPUT_DIR = 'out'

T_MAX = 50.
DT = 0.5

E = 1000.
NU = 0.45
ETA = 10000.

MU = mechanics.mu_from_E_nu(E, NU)
LAMBDA = mechanics.lambda_from_E_nu(E, NU)

CREEP_DISTANCE = 1.
CREEP_TIME = 10.

# def right_bc_function(x, t):
#     if t < CREEP_TIME:
#         return [CREEP_DISTANCE*t/CREEP_TIME, 0]
#     else:
#         return [CREEP_DISTANCE, 0]

def right_bc_function(x, t):
    if t < T_MAX/2.:
        return [CREEP_DISTANCE, 0]
    else:
        return [0, 0]


# def right_bc_function(x, t):
#     return [CREEP_DISTANCE, 0]

INITIAL_CONDITION = lambda x, t: [0., 0.]


class MassMatrix(matrix_assemblers.LinearElasticityMatrix):

    def set_param_viscosity(self, eta, plane_stress=False, plane_strain=False, thickness=None):
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
        self.set_param(matrix, plane_stress=plane_stress, plane_strain=plane_strain, thickness=thickness)


bottom_boundary = lambda x, t: x[1] <= 1e-12
top_boundary = lambda x, t: x[1] >= 1. -1e-12
left_boundary = lambda x, t: x[0] <= 1e-12
right_boundary = lambda x, t: x[0] >= 1.-1e-12

quadrature_degree = 1
function_size = 2
spatial_dimension = 2
element_degree = 1

if __name__=='__main__':
    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    a = matrix_assemblers.LinearElasticityMatrix(mesh, function_size)
    a.set_param_isotropic(LAMBDA, MU, plane_stress=True)

    m = MassMatrix(mesh, function_size)
    m.set_param_viscosity(ETA, plane_stress=True)

    b = vector_assemblers.ZeroVector(mesh, function_size)

    dirichlet_bcs = [
        DirichletBC(lambda x,t: [0.,0.], left_boundary),
        DirichletBC(right_bc_function, right_boundary),
    ]

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    t_array = time_integration.time_array(0., T_MAX, DT)

    u, f = time_integration.implicit_euler(
        m, a, b, dirichlet_bcs,
        INITIAL_CONDITION, t_array, out_prefix=os.path.join(OUTPUT_DIR, 'out_relaxation'))

    u.set_label('u')


