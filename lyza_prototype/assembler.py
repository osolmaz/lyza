import numpy as np
import logging
import time

class Assembler:
    def __init__(self, mesh, function_size, domain=None, quantity_dict={}):
        self.mesh = mesh
        self.function_size = function_size
        self.domain = domain
        self.quantity_dict=quantity_dict

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*function_size+i for i in range(function_size)])

        self.cell_dofs = []
        for c in self.mesh.cells:
            dofmap = []
            for n in c.nodes:
                node_dofs = [n.idx*self.function_size+i for i in range(self.function_size)]
                dofmap += self.node_dofs[n.idx]
            self.cell_dofs.append(dofmap)


class MatrixAssembler(Assembler):
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

class VectorAssembler(Assembler):
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

