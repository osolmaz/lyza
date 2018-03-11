from lyza_prototype.cell import Cell
from lyza_prototype.elements import HexElement1, QuadElement1, LineElement1
from lyza_prototype.quadrature_point import QuadraturePoint
from math import sqrt
import itertools

quad_weights_0 = [0.]
quad_coors_0 = [2.]

quad_weights_1 = [1., 1.]
quad_coors_1 = [-1./sqrt(3), 1./sqrt(3)]

quad_weights_2 = [5./9., 8./9., 5./9.]
quad_coors_2 = [-sqrt(3./5), 0., sqrt(3./5)]


class Hex(Cell):

    def get_finite_element(self, function_space, element_degree, quadrature_degree):
        if element_degree == 1:
            return HexElement1(
                self.nodes,
                self,
                function_space,
                quadrature_degree)
        else:
            raise Exception('Invalid element degree: %d'%elem_degree)


    def get_quad_points(self, quadrature_degree):
        if quadrature_degree == 0:
            weight_perm = list(itertools.product(quad_weights_0, quad_weights_0, quad_weights_0))
            quad_coors = list(itertools.product(quad_coors_0, quad_coors_0, quad_coors_0))
        elif quadrature_degree == 1:
            weight_perm = list(itertools.product(quad_weights_1, quad_weights_1, quad_weights_1))
            quad_coors = list(itertools.product(quad_coors_1, quad_coors_1, quad_coors_1))
        elif quadrature_degree == 2:
            weight_perm = list(itertools.product(quad_weights_2, quad_weights_2, quad_weights_2))
            quad_coors = list(itertools.product(quad_coors_2, quad_coors_2, quad_coors_2))
        else:
            raise Exception('Invalid quadrature degree')

        quad_weights = [i[0]*i[1]*i[2] for i in weight_perm]
        quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_points


class Quad(Cell):

    def get_finite_element(self, function_space, element_degree, quadrature_degree):
        if element_degree == 1:
            return QuadElement1(
                self.nodes,
                self,
                function_space,
                quadrature_degree)
        else:
            raise Exception('Invalid element degree: %d'%elem_degree)


    def get_quad_points(self, quadrature_degree):
        if quadrature_degree == 0:
            weight_perm = list(itertools.product(quad_weights_0, quad_weights_0))
            quad_coors = list(itertools.product(quad_coors_0, quad_coors_0))
        elif quadrature_degree == 1:
            weight_perm = list(itertools.product(quad_weights_1, quad_weights_1))
            quad_coors = list(itertools.product(quad_coors_1, quad_coors_1))
        elif quadrature_degree == 2:
            weight_perm = list(itertools.product(quad_weights_2, quad_weights_2))
            quad_coors = list(itertools.product(quad_coors_2, quad_coors_2))
        else:
            raise Exception('Invalid quadrature degree')

        quad_weights = [i[0]*i[1] for i in weight_perm]
        quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_points

class Line(Cell):
    def get_finite_element(self, function_space, element_degree, quadrature_degree):
        if element_degree == 1:
            return LineElement1(
                self.nodes,
                self,
                function_space,
                quadrature_degree)
        else:
            raise Exception('Invalid element degree: %d'%elem_degree)

    def get_quad_points(self, quadrature_degree):
        if quadrature_degree == 0:
            quad_weights = quad_weights_0
            quad_coors = quad_coors_0
        elif quadrature_degree == 1:
            quad_weights = quad_weights_1
            quad_coors = quad_coors_1
        elif quadrature_degree == 2:
            quad_weights = quad_weights_2
            quad_coors = quad_coors_2
        quad_coors = [[i] for i in quad_coors]

        quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_points
