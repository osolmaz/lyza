from lyza.cell import Cell
from math import sqrt
import itertools
import numpy as np

quad_weights_0 = [0.0]
quad_coors_0 = [2.0]

quad_weights_1 = [1.0, 1.0]
quad_coors_1 = [-1.0 / sqrt(3), 1.0 / sqrt(3)]

quad_weights_2 = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
quad_coors_2 = [-sqrt(3.0 / 5), 0.0, sqrt(3.0 / 5)]


class Hex(Cell):
    elem_dim = 3
    N = [
        lambda xi: 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]),
        lambda xi: 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]),
        lambda xi: 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]),
        lambda xi: 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]),
        lambda xi: 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]),
        lambda xi: 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]),
        lambda xi: 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]),
        lambda xi: 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]),
    ]

    Bhat = [
        lambda xi: np.array(
            [
                -0.125 * (1.0 - xi[1]) * (1.0 - xi[2]),
                -0.125 * (1.0 - xi[0]) * (1.0 - xi[2]),
                -0.125 * (1.0 - xi[0]) * (1.0 - xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                +0.125 * (1.0 - xi[1]) * (1.0 - xi[2]),
                -0.125 * (1.0 + xi[0]) * (1.0 - xi[2]),
                -0.125 * (1.0 + xi[0]) * (1.0 - xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                +0.125 * (1.0 + xi[1]) * (1.0 - xi[2]),
                +0.125 * (1.0 + xi[0]) * (1.0 - xi[2]),
                -0.125 * (1.0 + xi[0]) * (1.0 + xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                -0.125 * (1.0 + xi[1]) * (1.0 - xi[2]),
                +0.125 * (1.0 - xi[0]) * (1.0 - xi[2]),
                -0.125 * (1.0 - xi[0]) * (1.0 + xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                -0.125 * (1.0 - xi[1]) * (1.0 + xi[2]),
                -0.125 * (1.0 - xi[0]) * (1.0 + xi[2]),
                +0.125 * (1.0 - xi[0]) * (1.0 - xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                +0.125 * (1.0 - xi[1]) * (1.0 + xi[2]),
                -0.125 * (1.0 + xi[0]) * (1.0 + xi[2]),
                +0.125 * (1.0 + xi[0]) * (1.0 - xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                +0.125 * (1.0 + xi[1]) * (1.0 + xi[2]),
                +0.125 * (1.0 + xi[0]) * (1.0 + xi[2]),
                +0.125 * (1.0 + xi[0]) * (1.0 + xi[1]),
            ]
        ),
        lambda xi: np.array(
            [
                -0.125 * (1.0 + xi[1]) * (1.0 + xi[2]),
                +0.125 * (1.0 - xi[0]) * (1.0 + xi[2]),
                +0.125 * (1.0 - xi[0]) * (1.0 + xi[1]),
            ]
        ),
    ]

    def get_quad_points(self, quadrature_degree):
        if quadrature_degree == 0:
            weight_perm = list(
                itertools.product(quad_weights_0, quad_weights_0, quad_weights_0)
            )
            quad_coors = list(
                itertools.product(quad_coors_0, quad_coors_0, quad_coors_0)
            )
        elif quadrature_degree == 1:
            weight_perm = list(
                itertools.product(quad_weights_1, quad_weights_1, quad_weights_1)
            )
            quad_coors = list(
                itertools.product(quad_coors_1, quad_coors_1, quad_coors_1)
            )
        elif quadrature_degree == 2:
            weight_perm = list(
                itertools.product(quad_weights_2, quad_weights_2, quad_weights_2)
            )
            quad_coors = list(
                itertools.product(quad_coors_2, quad_coors_2, quad_coors_2)
            )
        else:
            raise Exception("Invalid quadrature degree")

        quad_coors = [np.array(list(i)).reshape(3, 1) for i in quad_coors]
        quad_weights = [
            np.array([[i[0] * i[1] * i[2]]]).reshape(1, 1) for i in weight_perm
        ]
        # quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_weights, quad_coors


class Quad(Cell):
    elem_dim = 2
    N = [
        lambda xi: 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]),
        lambda xi: 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]),
        lambda xi: 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]),
        lambda xi: 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]),
    ]

    Bhat = [
        lambda xi: np.array([-0.25 * (1.0 - xi[1]), -0.25 * (1.0 - xi[0])]),
        lambda xi: np.array([+0.25 * (1.0 - xi[1]), -0.25 * (1.0 + xi[0])]),
        lambda xi: np.array([+0.25 * (1.0 + xi[1]), +0.25 * (1.0 + xi[0])]),
        lambda xi: np.array([-0.25 * (1.0 + xi[1]), +0.25 * (1.0 - xi[0])]),
    ]

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
            raise Exception("Invalid quadrature degree")

        quad_coors = [np.array(list(i) + [0.0]).reshape(3, 1) for i in quad_coors]
        quad_weights = [np.array([i[0] * i[1]]).reshape(1, 1) for i in weight_perm]
        # quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_weights, quad_coors
        # return quad_points


class Line(Cell):
    elem_dim = 1
    N = [
        lambda xi: 0.5 * (1.0 + xi[0]),
        lambda xi: 0.5 * (1.0 - xi[0]),
    ]

    Bhat = [
        lambda xi: np.array([0.5]),
        lambda xi: np.array([-0.5]),
    ]

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

        quad_coors = [np.array([i, 0.0, 0.0]).reshape(3, 1) for i in quad_coors]
        quad_weights = [np.array([i]).reshape(1, 1) for i in quad_weights]
        # quad_points = [QuadraturePoint(i, j) for i,j in zip(quad_coors, quad_weights)]

        return quad_weights, quad_coors
        # return quad_points
