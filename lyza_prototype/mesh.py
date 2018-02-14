from lyza_prototype.node import Node

class Mesh:

    def __init__(self):
        self.nodes = []
        self.cells = []
        self.boundary_cells = []

        self.construct_mesh()

        for idx, n in enumerate(self.nodes):
            n.idx = idx

        for idx, c in enumerate(self.cells):
            c.idx = idx


    def construct_mesh():
        pass

    def add_node(self, coors):
        self.nodes.append(Node(coors, len(self.nodes)))

    def add_cell(self, cell):
        self.cells.append(cell)

    def get_n_nodes(self):
        return len(self.nodes)

