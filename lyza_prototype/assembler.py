import numpy as np
import logging
import time
from lyza_prototype.cell_iterator import CellIterator

class Assembler(CellIterator):
    def assemble(self):
        raise Exception('Do not use base class')

    def __add__(self, a):
        if isinstance(a, Assembler):
            return AggregateAssembler([self, a])
        elif isinstance(a, AggregateAssembler):
            return AggregateAssembler([self]+a.assemblers)
        else:
            raise Exception('Cannot add types')


class AggregateAssembler(Assembler):
    def __init__(self, assemblers):
        self.assemblers = assemblers

    def assemble(self):
        return sum([i.assemble() for i in self.assemblers])

    def __add__(self, a):
        if isinstance(a, Assembler):
            return AggregateAssembler(self.assemblers+[a])
        elif isinstance(a, AggregateAssembler):
            return AggregateAssembler(self.assemblers+a.assemblers)
        else:
            raise Exception('Cannot add types')


class MatrixAssembler(Assembler):
    def calculate_element_matrix(self, cell):
        raise Exception('Do not use base class')

    def assemble(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = np.zeros((n_dofs, n_dofs))

        logging.debug('Beginning to assemble matrix')
        start_time = time.time()

        for idx, cell in enumerate(self.mesh.cells):
            if not self.domain.is_subset(cell): continue

            elem_matrix = self.calculate_element_matrix(cell)
            dofmap = self.cell_dofs[idx]

            result[np.ix_(dofmap,dofmap)] += elem_matrix

            # for i, I in enumerate(dofmap):
            #     for j, J in enumerate(dofmap):
            #         result[I, J] += elem_matrix[i,j]

            # print(result[0:4,0:4])
        logging.debug('Matrix assembled in %f sec'%(time.time()-start_time))

        return result

class VectorAssembler(Assembler):
    def calculate_element_vector(self, cell):
        raise Exception('Do not use base class')

    def assemble(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = np.zeros((n_dofs, 1))

        logging.debug('Beginning to assemble vector')
        start_time = time.time()

        for idx, cell in enumerate(self.mesh.cells):
            if not self.domain.is_subset(cell): continue

            elem_vector = self.calculate_element_vector(cell)
            # print(elem_vector)
            dofmap = self.cell_dofs[idx]

            result[dofmap] += elem_vector

            # for i, I in enumerate(dofmap):
                    # result[I] += elem_vector[i]

        # import ipdb; ipdb.set_trace()
        logging.debug('Vector assembled in %f sec'%(time.time()-start_time))

        return result

