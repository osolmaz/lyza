from lyza_prototype.cells import Quad, Hex
from lyza_prototype.mesh import Mesh

class Cantilever3D(Mesh):
    def __init__(self, resolution, length, horizontal_width, vertical_width):

        self.resolution = resolution
        self.length = length
        self.horizontal_width = horizontal_width
        self.vertical_width = vertical_width

        super().__init__()

    def construct_mesh(self):
        cell_x_len = self.length/self.resolution

        for x in range(self.resolution+1):
            point_0 = [x*cell_x_len, 0., 0.]
            point_3 = [x*cell_x_len, self.horizontal_width, 0.]
            point_4 = [x*cell_x_len, self.horizontal_width, self.vertical_width]
            point_7 = [x*cell_x_len, 0., self.vertical_width]

            self.add_node(point_0)
            self.add_node(point_3)
            self.add_node(point_4)
            self.add_node(point_7)

        for i in range(self.resolution):
            n0 = self.nodes[i*4]
            n1 = self.nodes[(i+1)*4]
            n2 = self.nodes[(i+1)*4+1]
            n3 = self.nodes[i*4+1]
            n4 = self.nodes[i*4+3]
            n5 = self.nodes[(i+1)*4+3]
            n6 = self.nodes[(i+1)*4+2]
            n7 = self.nodes[i*4+2]

            self.add_cell(Hex([n0,n1,n2,n3,n4,n5,n6,n7]))

