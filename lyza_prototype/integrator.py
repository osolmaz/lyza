import numpy as np
import logging
import time

class Integrator:
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

    def integrate(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = 0.

        for idx, cell in enumerate(self.mesh.cells):
            if self.domain:
                pass
            else:
                if cell.is_boundary: continue

            elem_value = self.calculate_element_integral(cell)
            result += elem_value

        return result

