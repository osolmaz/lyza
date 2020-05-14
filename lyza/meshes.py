from math import pi as pi_val
from lyza.mesh import Mesh
from lyza.cells import Hex, Quad, Line
import copy


def locate_midpoint(coor1, coor2, percent):
    return [
        coor1[0] + (coor2[0] - coor1[0]) * percent,
        coor1[1] + (coor2[1] - coor1[1]) * percent,
    ]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # import ipdb; ipdb.set_trace()
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


class QuadMesh(Mesh):
    def __init__(self, resolution_x, resolution_y, p0, p1, p2, p3):
        self.res_x = resolution_x
        self.res_y = resolution_y

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        super().__init__()

    def construct_mesh(self):

        for y in range(self.res_y + 1):
            for x in range(self.res_x + 1):
                point_down = locate_midpoint(self.p0, self.p1, x / self.res_x)
                point_up = locate_midpoint(self.p3, self.p2, x / self.res_x)

                point_left = locate_midpoint(self.p0, self.p3, y / self.res_y)
                point_right = locate_midpoint(self.p1, self.p2, y / self.res_y)

                coor = line_intersection(
                    (point_left, point_right), (point_down, point_up)
                )
                self.add_node(coor + [0.0])

        for y in range(self.res_y):
            for x in range(self.res_x):
                n0 = self.nodes[y * (self.res_x + 1) + x]
                n1 = self.nodes[y * (self.res_x + 1) + x + 1]
                n2 = self.nodes[(y + 1) * (self.res_x + 1) + x + 1]
                n3 = self.nodes[(y + 1) * (self.res_x + 1) + x]
                self.add_cell(Quad([n0, n1, n2, n3]))

        for y in range(self.res_y):
            for x in range(self.res_x + 1):
                n0 = self.nodes[y * (self.res_x + 1) + x]
                n1 = self.nodes[(y + 1) * (self.res_x + 1) + x]
                self.add_cell(Line([n0, n1], is_boundary=True))

        for y in range(self.res_y + 1):
            for x in range(self.res_x):
                n0 = self.nodes[y * (self.res_x + 1) + x]
                n1 = self.nodes[y * (self.res_x + 1) + x + 1]
                self.add_cell(Line([n0, n1], is_boundary=True))


class UnitSquareMesh(QuadMesh):
    def __init__(self, resolution_x, resolution_y):
        super().__init__(
            resolution_x, resolution_y, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        )


class Cantilever3D(Mesh):
    def __init__(self, resolution, length, horizontal_width, vertical_width):

        self.resolution = resolution
        self.length = length
        self.horizontal_width = horizontal_width
        self.vertical_width = vertical_width

        super().__init__()

    def construct_mesh(self):
        cell_x_len = self.length / self.resolution

        for x in range(self.resolution + 1):
            point_0 = [x * cell_x_len, 0.0, 0.0]
            point_3 = [x * cell_x_len, self.horizontal_width, 0.0]
            point_4 = [x * cell_x_len, self.horizontal_width, self.vertical_width]
            point_7 = [x * cell_x_len, 0.0, self.vertical_width]

            self.add_node(point_0)
            self.add_node(point_3)
            self.add_node(point_4)
            self.add_node(point_7)

        for i in range(self.resolution):
            n0 = self.nodes[i * 4]
            n1 = self.nodes[(i + 1) * 4]
            n2 = self.nodes[(i + 1) * 4 + 1]
            n3 = self.nodes[i * 4 + 1]
            n4 = self.nodes[i * 4 + 3]
            n5 = self.nodes[(i + 1) * 4 + 3]
            n6 = self.nodes[(i + 1) * 4 + 2]
            n7 = self.nodes[i * 4 + 2]

            self.add_cell(Hex([n0, n1, n2, n3, n4, n5, n6, n7]))
