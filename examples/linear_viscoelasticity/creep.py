import os
import numpy as np
from math import sqrt
from lyza import *

import logging
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = 'out'

SPATIAL_DIMENSION = 3
FUNCTION_SIZE = 3
QUADRATURE_DEGREE = 1

RESOLUTION = 15
LENGTH = 300.
HORIZONTAL_WIDTH = 20.
VERTICAL_WIDTH = 20.

T_MAX = 30.*60*60
DT = T_MAX/20

E = 110.
NU = 0.35
ETA = 4.0e6*2/3

# Lame parameters
# LAMBDA = mechanics.lambda_from_E_nu(E, NU)
KAPPA = mechanics.kappa_from_E_nu(E, NU)

MU0 = 0
# MU0 = mechanics.mu_from_E_nu(E, NU)
MU1 = mechanics.mu_from_E_nu(E, NU)

LOAD = 2.5

left_boundary = lambda x, t: x[0] <= 1e-12
load_position = lambda x, t: x[0] > LENGTH-1e-12

def deviatoric(A):
    dev = A - 1/3*np.trace(A)*np.eye(3)
    return dev


class Calculator(CellIterator):
    def init_quantities(self):
        self.mesh.quantities['SIG'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['CTENSOR'] = CellQuantity(self.mesh, (3, 3, 3, 3))
        self.mesh.quantities['ALPHA'] = CellQuantity(self.mesh, (3, 3))

    def iterate(self, cell):

        EPS_arr = self.mesh.quantities['EPS'].get_quantity(cell)
        ALPHAN_arr = self.mesh.quantities['ALPHAN'].get_quantity(cell)

        self.mesh.quantities['SIG'].reset_quantity_by_cell(cell)
        self.mesh.quantities['CTENSOR'].reset_quantity_by_cell(cell)
        self.mesh.quantities['ALPHA'].reset_quantity_by_cell(cell)

        for idx in range(len(EPS_arr)):
            EPS = EPS_arr[idx]
            ALPHAN = ALPHAN_arr[idx]

            SIG,CTENSOR,ALPHA = self.material(EPS, ALPHAN)

            self.mesh.quantities['SIG'].add_quantity_by_cell(cell, SIG)
            self.mesh.quantities['CTENSOR'].add_quantity_by_cell(cell, CTENSOR)
            self.mesh.quantities['ALPHA'].add_quantity_by_cell(cell, ALPHA)

    def material(self, eps, alpha_n):
        tr_eps = np.trace(eps)
        eps_dev = deviatoric(eps)
        I = np.eye(3)

        tau = ETA/2/MU1

        alpha = 1/(1+DT/tau)*(alpha_n + DT/tau*eps_dev)
        sigma = KAPPA*tr_eps*I + 2*MU0*eps_dev + 2*MU1*(eps_dev-alpha)
        C = KAPPA*mechanics.IDENTITY_DYADIC_IDENTITY + (2*MU0 + 2*MU1/(1+DT/tau))*mechanics.PROJECTION4

        return sigma, C, alpha

def update_function(mesh, u):
    projector = iterators.SymmetricGradientProjector(mesh, u.function_size)
    projector.set_param(u, 'EPS', SPATIAL_DIMENSION)
    projector.execute()

    calculator = Calculator(mesh, u.function_size)
    calculator.init_quantities()
    calculator.execute()

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mesh = meshes.Cantilever3D(RESOLUTION, LENGTH, HORIZONTAL_WIDTH, VERTICAL_WIDTH)
    mesh.set_quadrature_degree(lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION)

    mesh.init_quantity('ALPHAN', (3,3))

    a = matrix_assemblers.InelasticityJacobianMatrix(mesh, FUNCTION_SIZE)
    b_res = vector_assemblers.InelasticityResidualVector(mesh, FUNCTION_SIZE)

    t = 0
    count = 0
    while t < T_MAX:
        b_load = vector_assemblers.PointLoadVector(mesh, FUNCTION_SIZE); b_load.set_param(load_position,[0.,0.,-LOAD])

        dirichlet_bcs = [DirichletBC(lambda x, t: [0.,0.,0.], left_boundary)]

        u, r = solver.nonlinear_solve(a, b_res+b_load, dirichlet_bcs, update_function=update_function)

        f = Function(mesh, FUNCTION_SIZE)
        f.set_vector(-1*b_res.assemble())

        # Voigt tensors
        voigt_converter = iterators.VoigtConverter(mesh, FUNCTION_SIZE)
        voigt_converter.set_param('SIG', 'SIGV')
        voigt_converter.execute()
        voigt_converter.set_param('ALPHA', 'ALPHAV')
        voigt_converter.execute()

        # Project to nodes for output
        sigma = mesh.quantities['SIGV'].get_function()
        eps_p = mesh.quantities['ALPHAV'].get_function()

        # Update history variables
        mesh.quantities['ALPHAN'] = mesh.quantities['ALPHA']

        ofile = VTKFile(os.path.join(OUTPUT_DIR, 'out_creep_%04d.vtk'%count))

        u.set_label('u')
        f.set_label('f')
        sigma.set_label('sigma')
        eps_p.set_label('eps_p')

        ofile.write(mesh, [u, f, sigma, eps_p])

        t += DT
        count += 1
