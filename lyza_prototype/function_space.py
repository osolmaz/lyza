from lyza_prototype.assembly import Assembly

class FunctionSpace:

    def __init__(self,
                 mesh,
                 function_dimension,
                 physical_dimension,
                 element_degree):

        self.mesh = mesh
        self.function_dimension = function_dimension
        self.physical_dimension = physical_dimension
        self.element_degree = element_degree
        # self.quadrature_degree = quadrature_degree

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*self.get_dimension()+i for i in range(self.get_dimension())])

        # self.elements = []
        # self.boundary_elements = []

        # for c in self.mesh.cells:
        #     self.elements.append(c.get_finite_element(self))

        # for c in self.mesh.boundary_cells:
        #     self.boundary_elements.append(c.get_finite_element(self))


    def get_dimension(self):
        return self.function_dimension


    def get_assembly(self, quadrature_degree, domain=None):

        result = []
        if domain:
            for c in self.mesh.cells:
                if domain.is_subset(c):
                    dofmap = []
                    for n in c.nodes:
                        node_dofs = [n.idx*self.function_dimension+i for i in range(self.function_dimension)]
                        dofmap += self.node_dofs[n.idx]

                    result.append(c.get_finite_element(
                        dofmap,
                        self,
                        self.element_degree,
                        quadrature_degree))
        else:
            for c in self.mesh.cells:
                if not c.is_boundary:
                    dofmap = []
                    for n in c.nodes:
                        node_dofs = [n.idx*self.function_dimension+i for i in range(self.function_dimension)]
                        dofmap += self.node_dofs[n.idx]
                    result.append(c.get_finite_element(
                        self,
                        self.element_degree,
                        quadrature_degree))



        return Assembly(result)

    def get_system_size(self):
        return self.mesh.get_n_nodes()*self.get_dimension()
