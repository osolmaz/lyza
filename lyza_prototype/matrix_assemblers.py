import itertools
from lyza_prototype.assembler import MatrixAssembler
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
