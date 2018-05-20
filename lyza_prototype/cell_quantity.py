import numpy as np
import logging

class CellQuantity:
    def __init__(self, mesh, quantity_shape):
        self.mesh = mesh
        self.quantity_shape = quantity_shape

        self.quantity_array_list = []
        self.quantity_array_dict = {}

        for cell in mesh.cells:
            array = []
            # for i in range(n_quantity_map(cell)):
            #     array.append(np.zeros(quantity_shape))

            self.quantity_array_list.append(array)
            self.quantity_array_dict[cell] = array

    def add_quantity_by_cell_idx(self, cell_idx, quantity_matrix):
        if quantity_matrix.shape != self.quantity_shape:
            logging.debug('Array shape %s does not match quantity shape %s'%(
                quantity_matrix.shape, self.quantity_shape))
            # raise Exception('Array shape %s does not match quantity shape %s'%(
            #     quantity_matrix.shape, self.quantity_shape))

        self.quantity_array_list[cell_idx].append(quantity_matrix)


    def get_quantity(self, cell):
        return self.quantity_array_dict[cell]

    def get_quantity_by_idx(self, cell_idx):
        return self.quantity_array_list[cell_idx]

