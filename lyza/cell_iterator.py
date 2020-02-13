import numpy as np
import logging
import time
from lyza.domain import DefaultDomain

flatten = lambda l: [item for sublist in l for item in sublist]

class CellIterator:
    def __init__(self, mesh, function_size, dof_ordering=None, domain=DefaultDomain()):
        self.mesh = mesh
        self.function_size = function_size
        self.domain = domain

        self.param = {}

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*function_size+i for i in range(function_size)])

        self.cell_dofs = []
        for c in self.mesh.cells:
            # for n in c.nodes:
                # node_dofs = [n.idx*self.function_size+i for i in range(self.function_size)]
                # dofmap += self.node_dofs[n.idx]

            node_dofs = [self.node_dofs[n.idx] for n in c.nodes]
            if not dof_ordering:
                dofmap = flatten(node_dofs)
            else:
                dofmap = []
                for indices in dof_ordering:
                    for idx in indices:
                        for dofs in node_dofs:
                            dofmap.append(dofs[idx])

            self.cell_dofs.append(dofmap)

    def set_param(self, param_dict):
        for key, value in param_dict.items():
            self.param[key] = value

    def set_time(self, time):
        self.time = time

    def execute(self):
        logging.debug('Beginning to assemble matrix')
        start_time = time.time()

        for idx, cell in enumerate(self.mesh.cells):
            if not self.domain.is_subset(cell): continue

            self.iterate(cell)
            dofmap = self.cell_dofs[idx]

        logging.debug('Cell iterator finished in %f sec'%(time.time()-start_time))


    def iterate(self, cell):
        raise Exception('Do not use base class')
