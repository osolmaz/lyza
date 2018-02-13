import itertools
from lyza_prototype.finite_element import FiniteElement
import numpy as np
from math import sqrt



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

