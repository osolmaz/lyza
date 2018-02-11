from math import sqrt, cos, sin, pi
import numpy as np
import itertools


class Cell:

    def __init__(self, nodes, label=None):
        self.label = label
        self.nodes = nodes
        self.n_node = len(self.nodes)

    def get_finite_element(self, function_space):
        raise Exception('Do not use the base class')

