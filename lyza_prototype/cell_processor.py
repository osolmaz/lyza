import numpy as np
import logging
import time

class CellProcessor:
    def __init__(self, mesh, function_size, domain=None):
        self.mesh = mesh
        self.function_size = function_size
        self.domain = domain

        self.param = {}

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

    def set_param(self, param_dict):
        for key, value in param_dict.items():
            self.param[key] = value
