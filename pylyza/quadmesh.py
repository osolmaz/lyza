from math import pi as pi_val
from pylyza.mesh import Mesh
from pylyza.element import QuadElement, LineElement
import copy

def locate_midpoint(coor1, coor2, percent):
    return [
        coor1[0] + (coor2[0]-coor1[0])*percent,
        coor1[1] + (coor2[1]-coor1[1])*percent,
    ]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # import ipdb; ipdb.set_trace()
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

class QuadMesh(Mesh):
    # def __init__(self, param):
    #     self.param = param
    #     super().__init__()

    def construct_mesh(self):
        # g1 = self.param[2]
        res_y = self.param['resolution_y']
        res_x = self.param['resolution_x']

        p0 = self.param['p0']
        p1 = self.param['p1']
        p2 = self.param['p2']
        p3 = self.param['p3']

        for y in range(res_y+1):
            for x in range(res_x+1):
                point_down = locate_midpoint(p0, p1, x/res_x)
                point_up = locate_midpoint(p3, p2, x/res_x)

                point_left = locate_midpoint(p0, p3, y/res_y)
                point_right = locate_midpoint(p1, p2, y/res_y)

                coor = line_intersection((point_left, point_right), (point_down, point_up))
                self.add_node(coor)

        for y in range(res_y):
            for x in range(res_x):
                n0 = self.nodes[y*(res_x+1) + x]
                n1 = self.nodes[y*(res_x+1) + x + 1]
                n2 = self.nodes[(y+1)*(res_x+1) + x + 1]
                n3 = self.nodes[(y+1)*(res_x+1) + x]
                self.add_elem(QuadElement([n0,n1,n2,n3], self.param))

        for y in range(res_y):
            for x in range(res_x+1):
                n0 = self.nodes[y*(res_x+1) + x]
                n1 = self.nodes[(y+1)*(res_x+1) + x]
                self.add_edge(LineElement([n0,n1], self.param))

        for y in range(res_y+1):
            for x in range(res_x):
                n0 = self.nodes[y*(res_x+1) + x]
                n1 = self.nodes[y*(res_x+1) + x + 1]
                self.add_edge(LineElement([n0,n1], self.param))




