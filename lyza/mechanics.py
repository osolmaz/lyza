import numpy as np
import itertools
from lyza.cell_quantity import CellQuantity


def mu_from_E_nu(E, nu):
    return E / (1.0 + nu) / 2.0


def lambda_from_E_nu(E, nu):
    return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)


def E_from_lambda_mu(lambda_, mu):
    return mu * (3.0 * lambda_ + 2.0 * mu) / (lambda_ + mu)


def nu_from_lambda_mu(lambda_, mu):
    return lambda_ / 2.0 / (lambda_ + mu)


def kappa_from_lambda_mu(lambda_, mu):
    return lambda_ + 2.0 / 3.0 * mu


def kappa_from_E_nu(E, nu):
    return E / 3.0 / (1.0 - 2.0 * nu)


VOIGT_INDEX_MAP3 = [[0, 3, 5], [3, 1, 4], [5, 4, 2]]

VOIGT_INDEX_MAP2 = [[0, 2], [2, 1]]

INVERSE_VOIGT_INDEX_MAP3 = [
    [0, 0],
    [1, 1],
    [2, 2],
    [0, 1],
    [1, 2],
    [0, 2],
]

INVERSE_VOIGT_INDEX_MAP2 = [
    [0, 0],
    [1, 1],
    [0, 1],
]

IDENTITY = np.identity(3)

IDENTITY4 = 0.5 * (
    np.einsum("ik,jl-> ijkl", IDENTITY, IDENTITY)
    + np.einsum("il,jk-> ijkl", IDENTITY, IDENTITY)
)

IDENTITY_DYADIC_IDENTITY = np.einsum("ij,kl-> ijkl", IDENTITY, IDENTITY)
PROJECTION4 = IDENTITY4 - (1 / 3) * IDENTITY_DYADIC_IDENTITY

IDENTITY4_VOIGT = np.array(
    [
        [1.0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1 / 2, 0, 0],
        [0, 0, 0, 0, 1 / 2, 0],
        [0, 0, 0, 0, 0, 1 / 2],
    ]
)

IDENTITY_DYADIC_IDENTITY_VOIGT = np.array(
    [
        [1.0, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

PROJECTION4_VOIGT = np.array(
    [
        [2 / 3, -1 / 3, -1 / 3, 0, 0, 0],
        [-1 / 3, 2 / 3, -1 / 3, 0, 0, 0],
        [-1 / 3, -1 / 3, 2 / 3, 0, 0, 0],
        [0, 0, 0, 1 / 2, 0, 0],
        [0, 0, 0, 0, 1 / 2, 0],
        [0, 0, 0, 0, 0, 1 / 2],
    ]
)


def voigt2(matrix):
    "Voigt a 2nd order tensor (input is a 3x3 or 2x2 matrix)"
    if matrix.shape == (3, 3):
        result = np.zeros((6, 1))
        voigt_index_map = VOIGT_INDEX_MAP3
    elif matrix.shape == (2, 2):
        result = np.zeros((3, 1))
        voigt_index_map = VOIGT_INDEX_MAP2
    else:
        raise Exception()

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        result[voigt_index_map[i][j]] = matrix[i, j]
    return result


def unvoigt4(C):
    "Unvoigt a 4th order tensor (input is a 6x6 or 3x3 matrix)"
    if C.shape == (6, 6):
        result = np.zeros((3, 3, 3, 3))
        index_map = INVERSE_VOIGT_INDEX_MAP3
        size = 6
    elif C.shape == (3, 3):
        result = np.zeros((2, 2, 2, 2))
        index_map = INVERSE_VOIGT_INDEX_MAP2
        size = 3
    else:
        raise Exception()

    for i in range(size):
        for j in range(size):
            a, b = index_map[i]
            c, d = index_map[j]

            result[a, b, c, d] = C[i, j]
            result[b, a, c, d] = C[i, j]
            result[a, b, d, c] = C[i, j]
            result[b, a, d, c] = C[i, j]
            # result[c,d,a,b] = C[i,j]

    return result


class ElasticityBase:
    def to_voigt(self, matrix):
        return voigt2(matrix)

    def set_param_isotropic(
        self, lambda_, mu, plane_stress=False, plane_strain=False, thickness=None
    ):
        C = np.array(
            [
                [lambda_ + 2 * mu, lambda_, lambda_, 0, 0, 0],
                [lambda_, lambda_ + 2 * mu, lambda_, 0, 0, 0],
                [lambda_, lambda_, lambda_ + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )
        self.set_param(
            C, plane_stress=plane_stress, plane_strain=plane_strain, thickness=thickness
        )

    def init_stress_quantity(
        self, spatial_dimension, stress_key="SIG", stress_voigt_key="SIGV"
    ):
        if spatial_dimension == 2:
            self.mesh.quantities[stress_key] = CellQuantity(self.mesh, (2, 2))
            self.mesh.quantities[stress_voigt_key] = CellQuantity(self.mesh, (3, 1))
        elif spatial_dimension == 3:
            self.mesh.quantities[stress_key] = CellQuantity(self.mesh, (3, 3))
            self.mesh.quantities[stress_voigt_key] = CellQuantity(self.mesh, (6, 1))

        self.stress_key = stress_key
        self.stress_voigt_key = stress_voigt_key

    def set_param(self, C, plane_stress=False, plane_strain=False, thickness=None):

        if plane_stress and plane_strain:
            raise Exception("Can be either plane stress or plane strain")

        # # Plane stress
        # self.C = E/(1-nu*nu)*np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1.-nu)/2.]])
        # self.index_map = [[0,2],[2,1]]

        # # Plane strain
        # self.C = E/(1.+nu)/(1-2.*nu)*np.array([[1.-nu,nu,0.],[nu,1.-nu,0.],[0.,0.,(1.-2.*nu)/2.]])
        # self.index_map = [[0,2],[2,1]]

        if plane_strain:
            self.C = np.array(
                [
                    [C[0, 0], C[0, 1], C[0, 3]],
                    [C[1, 0], C[1, 1], C[1, 3]],
                    [C[3, 0], C[3, 1], C[3, 3]],
                ]
            )
            self.index_map = [[0, 2], [2, 1]]
        elif plane_stress:
            self.C = np.array(
                [
                    [
                        (
                            C[0, 0] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[0, 0] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[0, 0] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[0, 0] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[0, 0] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[0, 0] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[0, 2] * C[2, 0] * C[4, 4] * C[5, 5]
                            + C[0, 2] * C[2, 0] * C[4, 5] * C[5, 4]
                            + C[0, 2] * C[2, 4] * C[4, 0] * C[5, 5]
                            - C[0, 2] * C[2, 4] * C[4, 5] * C[5, 0]
                            - C[0, 2] * C[2, 5] * C[4, 0] * C[5, 4]
                            + C[0, 2] * C[2, 5] * C[4, 4] * C[5, 0]
                            + C[0, 4] * C[2, 0] * C[4, 2] * C[5, 5]
                            - C[0, 4] * C[2, 0] * C[4, 5] * C[5, 2]
                            - C[0, 4] * C[2, 2] * C[4, 0] * C[5, 5]
                            + C[0, 4] * C[2, 2] * C[4, 5] * C[5, 0]
                            + C[0, 4] * C[2, 5] * C[4, 0] * C[5, 2]
                            - C[0, 4] * C[2, 5] * C[4, 2] * C[5, 0]
                            - C[0, 5] * C[2, 0] * C[4, 2] * C[5, 4]
                            + C[0, 5] * C[2, 0] * C[4, 4] * C[5, 2]
                            + C[0, 5] * C[2, 2] * C[4, 0] * C[5, 4]
                            - C[0, 5] * C[2, 2] * C[4, 4] * C[5, 0]
                            - C[0, 5] * C[2, 4] * C[4, 0] * C[5, 2]
                            + C[0, 5] * C[2, 4] * C[4, 2] * C[5, 0]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            C[0, 1] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[0, 1] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[0, 1] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[0, 1] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[0, 1] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[0, 1] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[0, 2] * C[2, 1] * C[4, 4] * C[5, 5]
                            + C[0, 2] * C[2, 1] * C[4, 5] * C[5, 4]
                            + C[0, 2] * C[2, 4] * C[4, 1] * C[5, 5]
                            - C[0, 2] * C[2, 4] * C[4, 5] * C[5, 1]
                            - C[0, 2] * C[2, 5] * C[4, 1] * C[5, 4]
                            + C[0, 2] * C[2, 5] * C[4, 4] * C[5, 1]
                            + C[0, 4] * C[2, 1] * C[4, 2] * C[5, 5]
                            - C[0, 4] * C[2, 1] * C[4, 5] * C[5, 2]
                            - C[0, 4] * C[2, 2] * C[4, 1] * C[5, 5]
                            + C[0, 4] * C[2, 2] * C[4, 5] * C[5, 1]
                            + C[0, 4] * C[2, 5] * C[4, 1] * C[5, 2]
                            - C[0, 4] * C[2, 5] * C[4, 2] * C[5, 1]
                            - C[0, 5] * C[2, 1] * C[4, 2] * C[5, 4]
                            + C[0, 5] * C[2, 1] * C[4, 4] * C[5, 2]
                            + C[0, 5] * C[2, 2] * C[4, 1] * C[5, 4]
                            - C[0, 5] * C[2, 2] * C[4, 4] * C[5, 1]
                            - C[0, 5] * C[2, 4] * C[4, 1] * C[5, 2]
                            + C[0, 5] * C[2, 4] * C[4, 2] * C[5, 1]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            -C[0, 2] * C[2, 3] * C[4, 4] * C[5, 5]
                            + C[0, 2] * C[2, 3] * C[4, 5] * C[5, 4]
                            + C[0, 2] * C[2, 4] * C[4, 3] * C[5, 5]
                            - C[0, 2] * C[2, 4] * C[4, 5] * C[5, 3]
                            - C[0, 2] * C[2, 5] * C[4, 3] * C[5, 4]
                            + C[0, 2] * C[2, 5] * C[4, 4] * C[5, 3]
                            + C[0, 3] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[0, 3] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[0, 3] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[0, 3] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[0, 3] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[0, 3] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[0, 4] * C[2, 2] * C[4, 3] * C[5, 5]
                            + C[0, 4] * C[2, 2] * C[4, 5] * C[5, 3]
                            + C[0, 4] * C[2, 3] * C[4, 2] * C[5, 5]
                            - C[0, 4] * C[2, 3] * C[4, 5] * C[5, 2]
                            - C[0, 4] * C[2, 5] * C[4, 2] * C[5, 3]
                            + C[0, 4] * C[2, 5] * C[4, 3] * C[5, 2]
                            + C[0, 5] * C[2, 2] * C[4, 3] * C[5, 4]
                            - C[0, 5] * C[2, 2] * C[4, 4] * C[5, 3]
                            - C[0, 5] * C[2, 3] * C[4, 2] * C[5, 4]
                            + C[0, 5] * C[2, 3] * C[4, 4] * C[5, 2]
                            + C[0, 5] * C[2, 4] * C[4, 2] * C[5, 3]
                            - C[0, 5] * C[2, 4] * C[4, 3] * C[5, 2]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                    ],
                    [
                        (
                            C[1, 0] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[1, 0] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[1, 0] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[1, 0] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[1, 0] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[1, 0] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[1, 2] * C[2, 0] * C[4, 4] * C[5, 5]
                            + C[1, 2] * C[2, 0] * C[4, 5] * C[5, 4]
                            + C[1, 2] * C[2, 4] * C[4, 0] * C[5, 5]
                            - C[1, 2] * C[2, 4] * C[4, 5] * C[5, 0]
                            - C[1, 2] * C[2, 5] * C[4, 0] * C[5, 4]
                            + C[1, 2] * C[2, 5] * C[4, 4] * C[5, 0]
                            + C[1, 4] * C[2, 0] * C[4, 2] * C[5, 5]
                            - C[1, 4] * C[2, 0] * C[4, 5] * C[5, 2]
                            - C[1, 4] * C[2, 2] * C[4, 0] * C[5, 5]
                            + C[1, 4] * C[2, 2] * C[4, 5] * C[5, 0]
                            + C[1, 4] * C[2, 5] * C[4, 0] * C[5, 2]
                            - C[1, 4] * C[2, 5] * C[4, 2] * C[5, 0]
                            - C[1, 5] * C[2, 0] * C[4, 2] * C[5, 4]
                            + C[1, 5] * C[2, 0] * C[4, 4] * C[5, 2]
                            + C[1, 5] * C[2, 2] * C[4, 0] * C[5, 4]
                            - C[1, 5] * C[2, 2] * C[4, 4] * C[5, 0]
                            - C[1, 5] * C[2, 4] * C[4, 0] * C[5, 2]
                            + C[1, 5] * C[2, 4] * C[4, 2] * C[5, 0]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            C[1, 1] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[1, 1] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[1, 1] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[1, 1] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[1, 1] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[1, 1] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[1, 2] * C[2, 1] * C[4, 4] * C[5, 5]
                            + C[1, 2] * C[2, 1] * C[4, 5] * C[5, 4]
                            + C[1, 2] * C[2, 4] * C[4, 1] * C[5, 5]
                            - C[1, 2] * C[2, 4] * C[4, 5] * C[5, 1]
                            - C[1, 2] * C[2, 5] * C[4, 1] * C[5, 4]
                            + C[1, 2] * C[2, 5] * C[4, 4] * C[5, 1]
                            + C[1, 4] * C[2, 1] * C[4, 2] * C[5, 5]
                            - C[1, 4] * C[2, 1] * C[4, 5] * C[5, 2]
                            - C[1, 4] * C[2, 2] * C[4, 1] * C[5, 5]
                            + C[1, 4] * C[2, 2] * C[4, 5] * C[5, 1]
                            + C[1, 4] * C[2, 5] * C[4, 1] * C[5, 2]
                            - C[1, 4] * C[2, 5] * C[4, 2] * C[5, 1]
                            - C[1, 5] * C[2, 1] * C[4, 2] * C[5, 4]
                            + C[1, 5] * C[2, 1] * C[4, 4] * C[5, 2]
                            + C[1, 5] * C[2, 2] * C[4, 1] * C[5, 4]
                            - C[1, 5] * C[2, 2] * C[4, 4] * C[5, 1]
                            - C[1, 5] * C[2, 4] * C[4, 1] * C[5, 2]
                            + C[1, 5] * C[2, 4] * C[4, 2] * C[5, 1]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            -C[1, 2] * C[2, 3] * C[4, 4] * C[5, 5]
                            + C[1, 2] * C[2, 3] * C[4, 5] * C[5, 4]
                            + C[1, 2] * C[2, 4] * C[4, 3] * C[5, 5]
                            - C[1, 2] * C[2, 4] * C[4, 5] * C[5, 3]
                            - C[1, 2] * C[2, 5] * C[4, 3] * C[5, 4]
                            + C[1, 2] * C[2, 5] * C[4, 4] * C[5, 3]
                            + C[1, 3] * C[2, 2] * C[4, 4] * C[5, 5]
                            - C[1, 3] * C[2, 2] * C[4, 5] * C[5, 4]
                            - C[1, 3] * C[2, 4] * C[4, 2] * C[5, 5]
                            + C[1, 3] * C[2, 4] * C[4, 5] * C[5, 2]
                            + C[1, 3] * C[2, 5] * C[4, 2] * C[5, 4]
                            - C[1, 3] * C[2, 5] * C[4, 4] * C[5, 2]
                            - C[1, 4] * C[2, 2] * C[4, 3] * C[5, 5]
                            + C[1, 4] * C[2, 2] * C[4, 5] * C[5, 3]
                            + C[1, 4] * C[2, 3] * C[4, 2] * C[5, 5]
                            - C[1, 4] * C[2, 3] * C[4, 5] * C[5, 2]
                            - C[1, 4] * C[2, 5] * C[4, 2] * C[5, 3]
                            + C[1, 4] * C[2, 5] * C[4, 3] * C[5, 2]
                            + C[1, 5] * C[2, 2] * C[4, 3] * C[5, 4]
                            - C[1, 5] * C[2, 2] * C[4, 4] * C[5, 3]
                            - C[1, 5] * C[2, 3] * C[4, 2] * C[5, 4]
                            + C[1, 5] * C[2, 3] * C[4, 4] * C[5, 2]
                            + C[1, 5] * C[2, 4] * C[4, 2] * C[5, 3]
                            - C[1, 5] * C[2, 4] * C[4, 3] * C[5, 2]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                    ],
                    [
                        (
                            -C[2, 0] * C[3, 2] * C[4, 4] * C[5, 5]
                            + C[2, 0] * C[3, 2] * C[4, 5] * C[5, 4]
                            + C[2, 0] * C[3, 4] * C[4, 2] * C[5, 5]
                            - C[2, 0] * C[3, 4] * C[4, 5] * C[5, 2]
                            - C[2, 0] * C[3, 5] * C[4, 2] * C[5, 4]
                            + C[2, 0] * C[3, 5] * C[4, 4] * C[5, 2]
                            + C[2, 2] * C[3, 0] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[3, 0] * C[4, 5] * C[5, 4]
                            - C[2, 2] * C[3, 4] * C[4, 0] * C[5, 5]
                            + C[2, 2] * C[3, 4] * C[4, 5] * C[5, 0]
                            + C[2, 2] * C[3, 5] * C[4, 0] * C[5, 4]
                            - C[2, 2] * C[3, 5] * C[4, 4] * C[5, 0]
                            - C[2, 4] * C[3, 0] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[3, 0] * C[4, 5] * C[5, 2]
                            + C[2, 4] * C[3, 2] * C[4, 0] * C[5, 5]
                            - C[2, 4] * C[3, 2] * C[4, 5] * C[5, 0]
                            - C[2, 4] * C[3, 5] * C[4, 0] * C[5, 2]
                            + C[2, 4] * C[3, 5] * C[4, 2] * C[5, 0]
                            + C[2, 5] * C[3, 0] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[3, 0] * C[4, 4] * C[5, 2]
                            - C[2, 5] * C[3, 2] * C[4, 0] * C[5, 4]
                            + C[2, 5] * C[3, 2] * C[4, 4] * C[5, 0]
                            + C[2, 5] * C[3, 4] * C[4, 0] * C[5, 2]
                            - C[2, 5] * C[3, 4] * C[4, 2] * C[5, 0]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            -C[2, 1] * C[3, 2] * C[4, 4] * C[5, 5]
                            + C[2, 1] * C[3, 2] * C[4, 5] * C[5, 4]
                            + C[2, 1] * C[3, 4] * C[4, 2] * C[5, 5]
                            - C[2, 1] * C[3, 4] * C[4, 5] * C[5, 2]
                            - C[2, 1] * C[3, 5] * C[4, 2] * C[5, 4]
                            + C[2, 1] * C[3, 5] * C[4, 4] * C[5, 2]
                            + C[2, 2] * C[3, 1] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[3, 1] * C[4, 5] * C[5, 4]
                            - C[2, 2] * C[3, 4] * C[4, 1] * C[5, 5]
                            + C[2, 2] * C[3, 4] * C[4, 5] * C[5, 1]
                            + C[2, 2] * C[3, 5] * C[4, 1] * C[5, 4]
                            - C[2, 2] * C[3, 5] * C[4, 4] * C[5, 1]
                            - C[2, 4] * C[3, 1] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[3, 1] * C[4, 5] * C[5, 2]
                            + C[2, 4] * C[3, 2] * C[4, 1] * C[5, 5]
                            - C[2, 4] * C[3, 2] * C[4, 5] * C[5, 1]
                            - C[2, 4] * C[3, 5] * C[4, 1] * C[5, 2]
                            + C[2, 4] * C[3, 5] * C[4, 2] * C[5, 1]
                            + C[2, 5] * C[3, 1] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[3, 1] * C[4, 4] * C[5, 2]
                            - C[2, 5] * C[3, 2] * C[4, 1] * C[5, 4]
                            + C[2, 5] * C[3, 2] * C[4, 4] * C[5, 1]
                            + C[2, 5] * C[3, 4] * C[4, 1] * C[5, 2]
                            - C[2, 5] * C[3, 4] * C[4, 2] * C[5, 1]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                        (
                            C[2, 2] * C[3, 3] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[3, 3] * C[4, 5] * C[5, 4]
                            - C[2, 2] * C[3, 4] * C[4, 3] * C[5, 5]
                            + C[2, 2] * C[3, 4] * C[4, 5] * C[5, 3]
                            + C[2, 2] * C[3, 5] * C[4, 3] * C[5, 4]
                            - C[2, 2] * C[3, 5] * C[4, 4] * C[5, 3]
                            - C[2, 3] * C[3, 2] * C[4, 4] * C[5, 5]
                            + C[2, 3] * C[3, 2] * C[4, 5] * C[5, 4]
                            + C[2, 3] * C[3, 4] * C[4, 2] * C[5, 5]
                            - C[2, 3] * C[3, 4] * C[4, 5] * C[5, 2]
                            - C[2, 3] * C[3, 5] * C[4, 2] * C[5, 4]
                            + C[2, 3] * C[3, 5] * C[4, 4] * C[5, 2]
                            + C[2, 4] * C[3, 2] * C[4, 3] * C[5, 5]
                            - C[2, 4] * C[3, 2] * C[4, 5] * C[5, 3]
                            - C[2, 4] * C[3, 3] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[3, 3] * C[4, 5] * C[5, 2]
                            + C[2, 4] * C[3, 5] * C[4, 2] * C[5, 3]
                            - C[2, 4] * C[3, 5] * C[4, 3] * C[5, 2]
                            - C[2, 5] * C[3, 2] * C[4, 3] * C[5, 4]
                            + C[2, 5] * C[3, 2] * C[4, 4] * C[5, 3]
                            + C[2, 5] * C[3, 3] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[3, 3] * C[4, 4] * C[5, 2]
                            - C[2, 5] * C[3, 4] * C[4, 2] * C[5, 3]
                            + C[2, 5] * C[3, 4] * C[4, 3] * C[5, 2]
                        )
                        / (
                            C[2, 2] * C[4, 4] * C[5, 5]
                            - C[2, 2] * C[4, 5] * C[5, 4]
                            - C[2, 4] * C[4, 2] * C[5, 5]
                            + C[2, 4] * C[4, 5] * C[5, 2]
                            + C[2, 5] * C[4, 2] * C[5, 4]
                            - C[2, 5] * C[4, 4] * C[5, 2]
                        ),
                    ],
                ]
            )
            self.index_map = VOIGT_INDEX_MAP2
        else:
            self.C = C
            self.index_map = VOIGT_INDEX_MAP3

        # import ipdb; ipdb.set_trace()
        self.C_unvoigt = unvoigt4(self.C)

        self.thickness = thickness
