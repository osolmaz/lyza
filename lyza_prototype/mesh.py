from lyza_prototype.node import Node

class Mesh:

    def __init__(self):
        self.nodes = []
        self.cells = []
        self.boundary_cells = []

        self.construct_mesh()

    def construct_mesh():
        pass

    def add_node(self, coors):
        self.nodes.append(Node(coors, len(self.nodes)))

    def add_cell(self, elem):
        self.cells.append(elem)

    def add_boundary_cell(self, elem):
        self.boundary_cells.append(elem)

    def get_n_nodes(self):
        return len(self.nodes)

