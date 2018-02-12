import itertools
from lyza_prototype.finite_element import FiniteElement
import numpy as np
from math import sqrt

quad_weights_0 = [0.]
quad_points_0 = [2.]

quad_weights_1 = [1., 1.]
quad_points_1 = [-1./sqrt(3), 1./sqrt(3)]

quad_weights_2 = [5./9., 8./9., 5./9.]
quad_points_2 = [-sqrt(3./5), 0., sqrt(3./5)]


class QuadElement1(FiniteElement):
    elem_dim = 2
    N = [
        lambda xi: 0.25*(1.-xi[0])*(1.-xi[1]),
        lambda xi: 0.25*(1.+xi[0])*(1.-xi[1]),
        lambda xi: 0.25*(1.+xi[0])*(1.+xi[1]),
        lambda xi: 0.25*(1.-xi[0])*(1.+xi[1]),
    ]

    Bhat = [
        lambda xi: np.array([-0.25*(1.-xi[1]), -0.25*(1.-xi[0])]),
        lambda xi: np.array([+0.25*(1.-xi[1]), -0.25*(1.+xi[0])]),
        lambda xi: np.array([+0.25*(1.+xi[1]), +0.25*(1.+xi[0])]),
        lambda xi: np.array([-0.25*(1.+xi[1]), +0.25*(1.-xi[0])]),
    ]

    def set_quad_points(self):
        if self.function_space.quadrature_degree == 0:
            weight_perm = list(itertools.product(quad_weights_0, quad_weights_0))
            self.quad_points = list(itertools.product(quad_points_0, quad_points_0))
        elif self.function_space.quadrature_degree == 1:
            weight_perm = list(itertools.product(quad_weights_1, quad_weights_1))
            self.quad_points = list(itertools.product(quad_points_1, quad_points_1))
        elif self.function_space.quadrature_degree == 2:
            weight_perm = list(itertools.product(quad_weights_2, quad_weights_2))
            self.quad_points = list(itertools.product(quad_points_2, quad_points_2))
        else:
            raise Exception('Invalid quadrature degree')

        self.quad_weights = [i[0]*i[1] for i in weight_perm]

class LineElement1(FiniteElement):
    elem_dim = 1
    N = [
        lambda xi: 0.5*(1.+xi[0]),
        lambda xi: 0.5*(1.-xi[0]),
    ]

    Bhat = [
        lambda xi: np.array([0.5]),
        lambda xi: np.array([-0.5]),
    ]

    def set_quad_points(self):
        if self.function_space.quadrature_degree == 0:
            self.quad_weights = quad_weights_0
            self.quad_points = quad_points_0
        elif self.function_space.quadrature_degree == 1:
            self.quad_weights = quad_weights_1
            self.quad_points = quad_points_1
        elif self.function_space.quadrature_degree == 2:
            self.quad_weights = quad_weights_2
            self.quad_points = quad_points_2
        self.quad_points = [[i] for i in self.quad_points]
