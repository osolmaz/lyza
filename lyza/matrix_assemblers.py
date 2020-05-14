import itertools
from lyza.assembler import MatrixAssembler
from lyza.mechanics import ElasticityBase
import numpy as np


class PoissonMatrix(MatrixAssembler):
    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]

            K += np.einsum("ij, kj ->  ik", B, B) * DETJ * W

            # for I,J,i in itertools.product(
            #         range(n_node),
            #         range(n_node),
            #         range(B.shape[1])):

            #     K[I, J] += B[I,i]*B[J,i]*DETJ*W

        return K


class MassMatrix(MatrixAssembler):

    # def set_param(self, eta):
    #     self.eta = eta

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        N_arr = self.mesh.quantities["N"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        for idx in range(len(W_arr)):
            N = N_arr[idx][:, 0]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]

            K += np.einsum("i, j ->  ij", N, N) * DETJ * W

            # for I,J,i in itertools.product(
            #         range(n_node),
            #         range(n_node)):
            #     K[I, J] += N[I,0]*N[J,0]*DETJ*W

        return K


class LinearElasticityMatrix(ElasticityBase, MatrixAssembler):
    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]
            spatial_dim = B.shape[1]

            K_contrib = (
                np.einsum("ic, acbd, jd -> iajb", B, self.C_unvoigt, B) * DETJ * W
            )
            K_contrib = K_contrib.reshape(K.shape)
            K += K_contrib

            # for I,J,i,j,k,l in itertools.product(
            #         range(n_node),
            #         range(n_node),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim),
            #         range(spatial_dim)):
            #     alpha = I*spatial_dim + i
            #     beta = J*spatial_dim + j
            #     C_val = self.C[self.index_map[i][k], self.index_map[j][l]]
            #     K[alpha, beta] += B[I,k]*C_val*B[J,l]*DETJ*W

        if self.thickness:
            K *= self.thickness

        return K


class InelasticityJacobianMatrix(MatrixAssembler):
    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)
        CTENSOR_arr = self.mesh.quantities["CTENSOR"].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]
            CTENSOR = CTENSOR_arr[idx]

            K_contrib = np.einsum("ic, acbd, jd -> iajb", B, CTENSOR, B) * DETJ * W
            K_contrib = K_contrib.reshape(K.shape)
            K += K_contrib

        return K


class HyperelasticityJacobian(MatrixAssembler):
    identity = np.eye(3)

    def set_param(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        FINVT_arr = self.mesh.quantities["FINVT"].get_quantity(cell)
        BBAR_arr = self.mesh.quantities["BBAR"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        LCG_arr = self.mesh.quantities["LCG"].get_quantity(cell)
        TAU_arr = self.mesh.quantities["TAU"].get_quantity(cell)
        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        K = np.zeros((n_dof, n_dof))

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            n_node = B.shape[0]
            spatial_dim = B.shape[1]

            c_eul = self.lambda_ * np.einsum("ab,cd->abcd", b_, b_) + self.mu * (
                np.einsum("ac,bd->abcd", b_, b_) + np.einsum("ad,bc->abcd", b_, b_)
            )

            K_higher = np.einsum("ic, acbd, jd -> iajb", Bbar, c_eul, Bbar) * DETJ * W
            K_higher += (
                np.einsum("ie, ef, jf, ab -> iajb", Bbar, tau, Bbar, self.identity)
                * DETJ
                * W
            )
            K_higher = K_higher.reshape(n_node * spatial_dim, n_node * spatial_dim)
            K += K_higher

        return K
