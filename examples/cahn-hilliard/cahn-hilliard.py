from lyza import *
from math import *
import numpy as np
import os
import random

from lyza.partition_system import partition_matrix, partition_vector

import logging

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "out"

QUADRATURE_DEGREE = 1
FUNCTION_SIZE = 2
SPATIAL_DIMENSION = 2
DOF_ORDERING = [[0], [1]]

# RESOLUTION = 10
# RESOLUTION = 30
RESOLUTION = 50

DT = 5e-6
T_MAX = DT * 200

f = lambda c: 100.0 * c ** 2 * (1.0 - c) ** 2
dfdc = lambda c: 200.0 * c * (2.0 * c ** 2 - 3.0 * c + 1.0)
d2fdc2 = lambda c: 200.0 * (6.0 * c ** 2 - 6.0 * c + 1.0)

M = np.eye(2)
alpha = 0.01 * np.eye(2)

initial_distribution = lambda x, t: [0.63 + 0.02 * (0.5 - random.random()), 0.0]


class CahnHilliardJacobianMatrix(lyza.MatrixAssembler):
    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node

        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        N_arr = self.mesh.quantities["N"].get_quantity(cell)
        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        C_arr = self.mesh.quantities["C"].get_quantity(cell)
        MU_arr = self.mesh.quantities["MU"].get_quantity(cell)

        A = np.zeros((n_dof, n_dof))
        B_ = np.zeros((n_dof, n_dof))
        D = np.zeros((n_dof, n_dof))
        E = np.zeros((n_dof, n_dof))

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            N = N_arr[idx][:, 0]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]

            C = C_arr[idx][0, 0]
            MU = MU_arr[idx][0, 0]

            A += 1.0 / DT * np.einsum("i, j -> ij", N, N) * DETJ * W
            B_ += np.einsum("ab, lb, ia -> il", M, B, B) * DETJ * W
            D += (
                -np.einsum("ab, lb, ia -> il", alpha, B, B) * DETJ * W
                - d2fdc2(C) * np.einsum("i, j -> ij", N, N) * DETJ * W
            )
            E += np.einsum("i, j -> ij", N, N) * DETJ * W

        K = np.vstack([np.hstack([A, B_]), np.hstack([D, E])])

        return K


class CahnHilliardResidualVector(VectorAssembler):
    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node

        c = np.zeros((n_dof, 1))
        f = np.zeros((n_dof, 1))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        N_arr = self.mesh.quantities["N"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        CN_arr = self.mesh.quantities["CN"].get_quantity(cell)
        MUN_arr = self.mesh.quantities["MUN"].get_quantity(cell)

        C_arr = self.mesh.quantities["C"].get_quantity(cell)
        MU_arr = self.mesh.quantities["MU"].get_quantity(cell)

        GRADC_arr = self.mesh.quantities["GRADC"].get_quantity(cell)
        GRADMU_arr = self.mesh.quantities["GRADMU"].get_quantity(cell)

        for idx in range(len(W_arr)):
            N = N_arr[idx][:, 0]
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]

            C = C_arr[idx][0, 0]
            MU = MU_arr[idx][0, 0]
            GRADC = GRADC_arr[idx][0, :]
            GRADMU = GRADMU_arr[idx][0, :]

            CN = CN_arr[idx][0, 0]
            MUN = MUN_arr[idx][0, 0]

            c_contrib = (C - CN) / DT * N * DETJ * W + np.einsum(
                "ab,b,ia->i", M, GRADMU, B
            ) * DETJ * W
            c += c_contrib.reshape(c.shape)

            f_contrib = (
                -np.einsum("ab,b,ia->i", alpha, GRADC, B) * DETJ * W
                + (MU - dfdc(C)) * N * DETJ * W
            )
            f += f_contrib.reshape(f.shape)

        r = np.vstack([c, f])
        r *= -1

        return r


# bottom_boundary = lambda x, t: x[1] <= 1e-12
# top_boundary = lambda x, t: x[1] >= 1. -1e-12
# left_boundary = lambda x, t: x[0] <= 1e-12
# right_boundary = lambda x, t: x[0] >= 1.-1e-12
# perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])


def update_function(mesh, u):
    c = u.separate_components([0])
    mu = u.separate_components([1])

    project_c_mu(mesh, c, mu)

    projector = iterators.GradientProjector(mesh, 1)
    projector.set_param(c, "GRADC", SPATIAL_DIMENSION)
    projector.execute()
    projector.set_param(mu, "GRADMU", SPATIAL_DIMENSION)
    projector.execute()


def project_c_mu(mesh, c, mu):
    projector = iterators.Projector(mesh, 1)
    projector.set_param(c, "C")
    projector.execute()
    projector.set_param(mu, "MU")
    projector.execute()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.set_quadrature_degree(lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION)

    u = Function(mesh, FUNCTION_SIZE)
    u.set_analytic_solution(initial_distribution)
    update_function(mesh, u)

    mesh.quantities["CN"] = mesh.quantities["C"].copy()
    mesh.quantities["MUN"] = mesh.quantities["MU"].copy()

    a = CahnHilliardJacobianMatrix(mesh, FUNCTION_SIZE, dof_ordering=DOF_ORDERING)
    b = CahnHilliardResidualVector(mesh, FUNCTION_SIZE, dof_ordering=DOF_ORDERING)
    dirichlet_bcs = []

    t = 0
    count = 0
    while t < T_MAX:

        if not count == 0:
            u, r = solver.nonlinear_solve(
                a, b, dirichlet_bcs, update_function=update_function, initial=u
            )

        ofile = VTKFile(os.path.join(OUTPUT_DIR, "out_cahn_hilliard_%05d.vtk" % count))

        c = u.separate_components([0])
        mu = u.separate_components([1])

        project_c_mu(mesh, c, mu)

        mesh.quantities["CN"] = mesh.quantities["C"]
        mesh.quantities["MUN"] = mesh.quantities["MU"]

        c.set_label("c")
        mu.set_label("mu")

        ofile.write(mesh, [c, mu])

        t += DT
        count += 1
