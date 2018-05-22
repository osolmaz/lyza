import numpy as np
import logging
import time
from lyza_prototype.cell_iterator import CellIterator


class MatrixAssembler(CellIterator):
    def calculate_element_matrix(self, cell):
        raise Exception('Do not use base class')

    def assemble(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = np.zeros((n_dofs, n_dofs))

        logging.debug('Beginning to assemble matrix')
        start_time = time.time()

        for idx, cell in enumerate(self.mesh.cells):
            if self.domain:
                pass
            else:
                if cell.is_boundary: continue

            elem_matrix = self.calculate_element_matrix(cell)
            dofmap = self.cell_dofs[idx]

            for i, I in enumerate(dofmap):
                for j, J in enumerate(dofmap):
                    result[I, J] += elem_matrix[i,j]
            # print(result[0:4,0:4])
        logging.debug('Matrix assembled in %f sec'%(time.time()-start_time))

        return result

class VectorAssembler(CellIterator):
    def calculate_element_vector(self, cell):
        raise Exception('Do not use base class')

    def assemble(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = np.zeros((n_dofs, 1))

        logging.debug('Beginning to assemble vector')
        start_time = time.time()

        for idx, cell in enumerate(self.mesh.cells):
            if self.domain:
                pass
            else:
                if cell.is_boundary: continue

            elem_vector = self.calculate_element_vector(cell)
            dofmap = self.cell_dofs[idx]

            for i, I in enumerate(dofmap):
                    result[I] += elem_vector[i]

        logging.debug('Vector assembled in %f sec'%(time.time()-start_time))

        return result

