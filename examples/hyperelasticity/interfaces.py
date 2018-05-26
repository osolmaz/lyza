import numpy as np
from lyza_prototype.assembler import MatrixAssembler, VectorAssembler
from lyza_prototype.cell_iterator import CellIterator
import itertools
import time

IDENTITY = np.eye(3)

class HyperelasticityJacobian(MatrixAssembler):

    def set_param(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        FINVT_arr = self.mesh.quantities['FINVT'].get_quantity(cell)
        BBAR_arr = self.mesh.quantities['BBAR'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        LCG_arr = self.mesh.quantities['LCG'].get_quantity(cell)
        TAU_arr = self.mesh.quantities['TAU'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        K = np.zeros((n_dof,n_dof))

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            n_node = B.shape[0]
            spatial_dim = B.shape[1]

            c_eul = self.lambda_*np.einsum('ab,cd->abcd', b_, b_) \
                + self.mu*(
                    np.einsum('ac,bd->abcd',b_, b_)
                    +np.einsum('ad,bc->abcd',b_, b_)
                )

            K_higher = np.einsum('ic, acbd, jd -> iajb', Bbar, c_eul, Bbar)*DETJ*W
            K_higher += np.einsum('ie, ef, jf, ab -> iajb', Bbar, tau, Bbar, IDENTITY)*DETJ*W
            K_higher = K_higher.reshape(n_node*spatial_dim, n_node*spatial_dim)
            K += K_higher

        return K


class HyperelasticityResidual(VectorAssembler):

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        FINVT_arr = self.mesh.quantities['FINVT'].get_quantity(cell)
        BBAR_arr = self.mesh.quantities['BBAR'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        LCG_arr = self.mesh.quantities['LCG'].get_quantity(cell)
        TAU_arr = self.mesh.quantities['TAU'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        f = np.zeros((n_dof, 1))

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            n_node = B.shape[0]
            spatial_dim = B.shape[1]

            f_higher = -np.einsum('ab, ib -> ia', tau, Bbar)*DETJ*W
            f_higher = f_higher.reshape((n_node*spatial_dim,1))
            f += f_higher

        return f


