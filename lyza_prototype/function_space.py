
class FunctionSpace:

    def __init__(self,
                 mesh,
                 function_dimension,
                 physical_dimension,
                 element_degree,
                 quadrature_degree):

        self.mesh = mesh
        self.function_dimension = function_dimension
        self.physical_dimension = physical_dimension
        self.element_degree = element_degree
        self.quadrature_degree = quadrature_degree

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*self.get_dimension()+i for i in range(self.get_dimension())])


    def get_dimension(self):
        return self.function_dimension


    def get_finite_elements(self, domain=None):
        result = []
        if domain:
            for c in self.mesh.cells:
                if domain.is_subset(c, False):
                    result.append(c.get_finite_element(self))
            for c in self.mesh.boundary_cells:
                if domain.is_subset(c, True):
                    result.append(c.get_finite_element(self))
        else:
            for c in self.mesh.cells:
                result.append(c.get_finite_element(self))
        return result


    def get_system_size(self):
        return self.mesh.get_n_nodes()*self.get_dimension()
