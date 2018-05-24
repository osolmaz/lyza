from lyza_prototype.function import Function
import numpy as np

class FunctionSpace:

    def __init__(self,
                 mesh,
                 function_size,
                 spatial_dimension,
                 element_degree):

        self.mesh = mesh
        self.function_size = function_size
        self.spatial_dimension = spatial_dimension
        self.element_degree = element_degree

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*self.get_dimension()+i for i in range(self.get_dimension())])


    def get_dimension(self):
        return self.function_size


    def get_finite_elements(self, quadrature_degree, domain=None):

        result = []
        if domain:
            for c in self.mesh.cells:
                if domain.is_subset(c):
                    dofmap = []
                    for n in c.nodes:
                        node_dofs = [n.idx*self.function_size+i for i in range(self.function_size)]
                        dofmap += self.node_dofs[n.idx]

                    result.append(c.get_finite_element(
                        self,
                        self.element_degree,
                        quadrature_degree))
        else:
            for c in self.mesh.cells:
                if not c.is_boundary:
                    dofmap = []
                    for n in c.nodes:
                        node_dofs = [n.idx*self.function_size+i for i in range(self.function_size)]
                        dofmap += self.node_dofs[n.idx]
                    result.append(c.get_finite_element(
                        self,
                        self.element_degree,
                        quadrature_degree))


        return result

    def get_system_size(self):
        return self.mesh.get_n_nodes()*self.get_dimension()

