import os
import numpy as np
from math import sqrt
from lyza_prototype import *

import logging
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = 'out'

spatial_dimension = 3
function_size = 3
element_degree = 1
quadrature_degree = 1

RESOLUTION = 15
LENGTH = 300.
HORIZONTAL_WIDTH = 20.
VERTICAL_WIDTH = 20.

E = 200.
E_T = 20.
NU = 0.3

E_P = E*E_T/(E-E_T)

# Lame parameters
# LAMBDA = mechanics.lambda_from_E_nu(E, NU)
KAPPA = mechanics.kappa_from_E_nu(E, NU)
MU = mechanics.mu_from_E_nu(E, NU)

Y0 = 0.2 # yield stress
H_ISO = E_P

LOADS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
# LOADS = [5.0]

left_boundary = lambda x, t: x[0] <= 1e-12

load_position_left = lambda x, t: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12
load_position_right = lambda x, t: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12

def deviatoric(A):
    dev = A - 1/3*np.trace(A)*np.eye(3)
    return dev


class Calculator(CellIterator):
    def init_quantities(self):
        self.mesh.quantities['SIG'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['CTENSOR'] = CellQuantity(self.mesh, (3, 3, 3, 3))

        self.mesh.quantities['EPSP'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['ALPHA'] = CellQuantity(self.mesh, (1, 1))

    def iterate(self, cell):

        EPS_arr = self.mesh.quantities['EPS'].get_quantity(cell)
        EPSPN_arr = self.mesh.quantities['EPSPN'].get_quantity(cell)
        ALPHAN_arr = self.mesh.quantities['ALPHAN'].get_quantity(cell)

        self.mesh.quantities['SIG'].reset_quantity_by_cell(cell)
        self.mesh.quantities['CTENSOR'].reset_quantity_by_cell(cell)
        self.mesh.quantities['EPSP'].reset_quantity_by_cell(cell)
        self.mesh.quantities['ALPHA'].reset_quantity_by_cell(cell)

        for idx in range(len(EPS_arr)):
            EPS = EPS_arr[idx]
            EPSPN = EPSPN_arr[idx]
            ALPHAN = ALPHAN_arr[idx]

            SIG,CTENSOR,EPSP,ALPHA = self.material(EPS, EPSPN, ALPHAN)

            self.mesh.quantities['SIG'].add_quantity_by_cell(cell, SIG)
            self.mesh.quantities['CTENSOR'].add_quantity_by_cell(cell, CTENSOR)
            self.mesh.quantities['EPSP'].add_quantity_by_cell(cell, EPSP)
            self.mesh.quantities['ALPHA'].add_quantity_by_cell(cell, ALPHA)


    def material(self, eps, eps_p_n, alpha_n):
        tr_eps = np.trace(eps)
        eps_dev = deviatoric(eps)
        I = np.eye(3)

        sigma_dev_tr = 2.*MU*(eps_dev-eps_p_n)
        beta_tr = H_ISO*alpha_n

        # xi_tr = sigma_dev_tr - beta_tr
        xi_tr = sigma_dev_tr
        norm_xi_tr = np.linalg.norm(xi_tr)

        phi_tr = norm_xi_tr - sqrt(2/3)*(Y0+beta_tr)

        if phi_tr <= 0: # elastic step
            sigma_dev = sigma_dev_tr
            beta = beta_tr
            C_dev = 2*MU*mechanics.PROJECTION4
        else: # elastic-plastic step
            n = xi_tr/norm_xi_tr
            gamma = phi_tr/(2*MU+(2/3)*H_ISO)
            sigma_dev = sigma_dev_tr - 2*MU*gamma*n
            beta = beta_tr + H_ISO*gamma*sqrt(2/3)

            n_dyadic_n = np.einsum('ij,kl->ijkl', n, n)
            c1 = 1-1/(1+H_ISO/(3*MU))*phi_tr/norm_xi_tr
            c2 = 1/(1+H_ISO/(3*MU))*(1-phi_tr/norm_xi_tr)
            C_dev = 2*MU*c1*mechanics.PROJECTION4 - 2*MU*c2*n_dyadic_n

        sigma = KAPPA*tr_eps*I + sigma_dev
        C = KAPPA*mechanics.IDENTITY_DYADIC_IDENTITY + C_dev

        eps_p = eps_p_n + (sigma_dev_tr-sigma_dev)/(2*MU)
        alpha = beta/H_ISO

        # print(eps_p-eps_p.T)
        # print(eps_p_n)
        # print(phi_tr, norm_xi_tr)

        # Linear elastic
        # sigma = KAPPA*tr_eps*I + 2*MU*eps_dev
        # C = KAPPA*IdI + 2*MU*P

        # import ipdb; ipdb.set_trace()
        return sigma, C, eps_p, alpha

def update_function(mesh, u):
    projector = iterators.SymmetricGradientProjector(mesh, u.function_size)
    projector.set_param(u, 'EPS', spatial_dimension)
    projector.execute()

    calculator = Calculator(mesh, u.function_size)
    calculator.init_quantities()
    calculator.execute()

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mesh = meshes.Cantilever3D(RESOLUTION, LENGTH, HORIZONTAL_WIDTH, VERTICAL_WIDTH)
    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    mesh.init_quantity('EPSPN', (3,3))
    mesh.init_quantity('ALPHAN', (1,1))

    a = matrix_assemblers.InelasticityJacobianMatrix(mesh, function_size)
    b_res = vector_assemblers.InelasticityResidualVector(mesh, function_size)

    for idx, load in enumerate(LOADS):
        b_1 = vector_assemblers.PointLoadVector(mesh, function_size); b_1.set_param(load_position_left,[0.,0.,-load])
        b_2 = vector_assemblers.PointLoadVector(mesh, function_size); b_2.set_param(load_position_right,[0.,0.,-load])

        b_load = b_1 + b_2

        dirichlet_bcs = [DirichletBC(lambda x, t: [0.,0.,0.], left_boundary)]

        u, r = solver.nonlinear_solve(a, b_res+b_load, dirichlet_bcs, update_function=update_function)

        f = Function(mesh, function_size)
        f.set_vector(-1*b_res.assemble())

        # Voigt tensors
        voigt_converter = iterators.VoigtConverter(mesh, function_size)
        voigt_converter.set_param('SIG', 'SIGV')
        voigt_converter.execute()
        voigt_converter.set_param('EPSP', 'EPSPV')
        voigt_converter.execute()

        # Project to nodes for output
        sigma = mesh.quantities['SIGV'].get_function()
        eps_p = mesh.quantities['EPSPV'].get_function()
        alpha = mesh.quantities['ALPHA'].get_function()

        # Update history variables
        mesh.quantities['EPSPN'] = mesh.quantities['EPSP']
        mesh.quantities['ALPHAN'] = mesh.quantities['ALPHA']

        ofile = VTKFile(os.path.join(OUTPUT_DIR, 'out_plasticity_%04d.vtk'%idx))

        u.set_label('u')
        f.set_label('f')
        sigma.set_label('sigma')
        eps_p.set_label('eps_p')
        alpha.set_label('alpha')

        ofile.write(mesh, [u, f, sigma, eps_p, alpha])
