import numpy as np
import logging
from lyza_prototype.function import Function

class CellQuantity:
    def __init__(self, mesh, shape):
        self.mesh = mesh
        self.shape = shape

        self.quantity_array_list = []
        self.quantity_array_dict = {}

        for cell in mesh.cells:
            array = []
            # for i in range(n_quantity_map(cell)):
            #     array.append(np.zeros(shape))

            self.quantity_array_list.append(array)
            self.quantity_array_dict[cell] = array

    def add_quantity_by_cell_idx(self, cell_idx, quantity_matrix):
        if quantity_matrix.shape != self.shape:
            logging.debug('Array shape %s does not match quantity shape %s'%(
                quantity_matrix.shape, self.shape))
            # raise Exception('Array shape %s does not match quantity shape %s'%(
            #     quantity_matrix.shape, self.shape))

        self.quantity_array_list[cell_idx].append(quantity_matrix)

    def add_quantity_by_cell(self, cell, quantity_matrix):
        if quantity_matrix.shape != self.shape:
            logging.debug('Array shape %s does not match quantity shape %s'%(
                quantity_matrix.shape, self.shape))
            # raise Exception('Array shape %s does not match quantity shape %s'%(
            #     quantity_matrix.shape, self.shape))

        self.quantity_array_dict[cell].append(quantity_matrix)

    def reset_quantity_by_cell(self, cell):
        self.quantity_array_dict[cell] = []


    def get_quantity(self, cell):
        return self.quantity_array_dict[cell]

    def get_quantity_by_idx(self, cell_idx):
        return self.quantity_array_list[cell_idx]

    def get_function(self):
        # if function_space == 1:
        #     target_space = self.function_space_1
        # elif function_space == 2:
        #     target_space = self.function_space_2
        # else:
        #     raise Exception('Function space can be either 1 or 2')

        if self.shape[1] > 1:
            raise Exception('Projecting matrix quantities not yet implemented')

        function_size = self.shape[0]

        result = Function(self.mesh, function_size)

        n_dof = len(self.mesh.nodes)*function_size
        f = np.zeros((n_dof,1))
        w = np.zeros((n_dof,1))

        for cell in self.mesh.cells:
            # target_elem = interface.elements[function_space-1]

            # target_quantity = quantity_map(interface)

            for node_i, node in enumerate(cell.nodes):
                f_elem = self._projection_vector(cell, node_i)
                w_elem = self._projection_weight_vector(cell, node_i)
                dofs = result.node_dofs[node.idx]

                for dof_i, dof in enumerate(dofs):
                    f[dof] += f_elem[dof_i]
                    w[dof] += w_elem[dof_i]

        projected_values = f/w
        result.set_vector(projected_values)

        return result

    def _projection_vector(self, cell, node_idx):
        n_dof = self.shape[0]
        f = np.zeros((n_dof,1))

        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        arrays = self.get_quantity(cell)

        for idx in range(len(W_arr)):
            DETJ = DETJ_arr[idx][0,0]
            W = W_arr[idx][0,0]
            N = N_arr[idx]
            for i in range(n_dof):
                f[i] += arrays[idx][i,0]*N[node_idx,0]*DETJ*W

        return f

    def _projection_weight_vector(self, cell, node_idx):
        n_dof = self.shape[0]
        f = np.zeros((n_dof,1))

        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)
        W_arr = self.mesh.quantities['W'].get_quantity(cell)

        for idx in range(len(W_arr)):
            DETJ = DETJ_arr[idx][0,0]
            W = W_arr[idx][0,0]
            N = N_arr[idx]
            for i in range(n_dof):
                f[i] += N[node_idx,0]*DETJ*W

        return f
