from lyza_prototype.cell_iterator import CellIterator
from lyza_prototype.cell_quantity import CellQuantity
from lyza_prototype.elasticity import ElasticityBase
import numpy as np
import itertools



class Projector(CellIterator):
    def set_param(self, function, quantity_key):
        self.function = function
        self.quantity_key = quantity_key

        self.mesh.quantities[quantity_key] = CellQuantity(self.mesh, (function.function_size, 1))

    def iterate(self, cell):

        n_node = len(cell.nodes)
        N_arr = self.mesh.quantities['N'].get_quantity(cell)

        for idx in range(len(N_arr)):
            result = np.zeros((self.function.function_size,1))
            N = N_arr[idx]

            for I in range(n_node):
                val = self.function.get_node_val(cell.nodes[I].idx)

                for i in range(self.function.function_size):
                    result[i] += N[I]*val[i]

            self.mesh.quantities[self.quantity_key].add_quantity_by_cell(cell, result)


class GradientProjector(CellIterator):
    def set_param(self, function, quantity_key, spatial_dimension):
        self.function = function
        self.quantity_key = quantity_key
        self.spatial_dimension = spatial_dimension

        self.mesh.quantities[quantity_key] = CellQuantity(self.mesh, (function.function_size, spatial_dimension))

    def iterate(self, cell):

        n_node = len(cell.nodes)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)

        for idx in range(len(B_arr)):
            result = np.zeros((self.function.function_size,self.spatial_dimension))
            B = B_arr[idx]

            for I in range(n_node):
                val = self.function.get_node_val(cell.nodes[I].idx)

                for i in range(self.function.function_size):
                    for j in range(self.spatial_dimension):
                        result[i,j] += B[I,j]*val[i]

            self.mesh.quantities[self.quantity_key].add_quantity_by_cell(cell, result)

class SymmetricGradientProjector(GradientProjector):

    def iterate(self, cell):

        n_node = len(cell.nodes)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)

        for idx in range(len(B_arr)):
            result = np.zeros((self.function.function_size,self.spatial_dimension))
            B = B_arr[idx]

            for I in range(n_node):
                val = self.function.get_node_val(cell.nodes[I].idx)

                for i in range(self.function.function_size):
                    for j in range(self.spatial_dimension):
                        result[i,j] += B[I,j]*val[i]

            result = 0.5*(result+result.T)
            self.mesh.quantities[self.quantity_key].add_quantity_by_cell(cell, result)


class LinearStressCalculator(ElasticityBase, CellIterator):

    def iterate(self, cell):

        n_node = len(cell.nodes)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        EPS_arr = self.mesh.quantities['EPS'].get_quantity(cell)

        self.mesh.quantities[self.stress_key].reset_quantity_by_cell(cell)

        for idx in range(len(W_arr)):
            strain = EPS_arr[idx]

            # strain = (grad_u + grad_u.T)/2.
            stress = np.zeros(self.mesh.quantities[self.stress_key].shape)

            for i, j, k, l in itertools.product(
                    range(stress.shape[0]), range(stress.shape[1]),
                    range(strain.shape[0]), range(strain.shape[1])):
                stress[i,j] += self.C[self.index_map[i][j], self.index_map[k][l]]*strain[k,l]

            # strain_voigt = to_voigt(strain)
            stress_voigt = self.to_voigt(stress)

            self.mesh.quantities[self.stress_key].add_quantity_by_cell(cell, stress)
            self.mesh.quantities[self.stress_voigt_key].add_quantity_by_cell(cell, stress_voigt)
            # self.mesh.quantities[self.strain_key].add_quantity_by_cell(cell, stress)



