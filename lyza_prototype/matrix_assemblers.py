import itertools
from lyza_prototype.assembler import MatrixAssembler
from lyza_prototype.elasticity import ElasticityBase
import numpy as np

class PoissonMatrix(MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            for I,J,i in itertools.product(
                    range(n_node),
                    range(n_node),
                    range(B.shape[1])):

                K[I, J] += B[I,i]*B[J,i]*DETJ*W

        return K


class MassMatrix(MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        for idx in range(len(W_arr)):
            N = N_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            for I,J,i in itertools.product(
                    range(n_node),
                    range(n_node)):

                K[I, J] += N[I,0]*N[J,0]*DETJ*W

        return K


class LinearElasticity(ElasticityBase, MatrixAssembler):

    def calculate_element_matrix(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        K = np.zeros((n_dof,n_dof))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]
            spatial_dim = B.shape[1]

            for I,J,i,j,k,l in itertools.product(
                    range(n_node),
                    range(n_node),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = I*spatial_dim + i
                beta = J*spatial_dim + j

                C_val = self.C[self.index_map[i][k], self.index_map[j][l]]
                K[alpha, beta] += B[I,k]*C_val*B[J,l]*DETJ*W

        if self.thickness:
            K *= self.thickness

        return K
