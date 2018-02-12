
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

        self.elements = []
        self.boundary_elements = []

        for c in self.mesh.cells:
            self.elements.append(c.get_finite_element(self))

        for c in self.mesh.boundary_cells:
            self.boundary_elements.append(c.get_finite_element(self))


    def get_dimension(self):
        return self.function_dimension


    def get_finite_elements(self, domain=None):
        result = []
        if domain:
            for c, e in zip(self.mesh.cells, self.elements):
                if domain.is_subset(c, False):
                    result.append(e)
            for c, e in zip(self.mesh.boundary_cells, self.boundary_elements):
                if domain.is_subset(c, True):
                    result.append(e)
        else:
            for c, e in zip(self.mesh.cells, self.elements):
                result.append(e)

        return result

    def get_system_size(self):
        return self.mesh.get_n_nodes()*self.get_dimension()
