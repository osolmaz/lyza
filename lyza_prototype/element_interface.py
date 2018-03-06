

class ElementInterface:

    def __init__(self):
        pass

    def matrix(self):
        raise Exception('Undefined element interface')

    def vector(self):
        raise Exception('Undefined element interface')

    def evaluate(self):
        raise Exception('Undefined element interface')

    def set_elements(self, elements):
        if not len(elements) in [1,2]:
            raise Exception()

        self.elements = elements
        self.n_quad_point = self.elements[0].n_quad_point

        self.time = 0

    def set_time(self, t):
        self.time = t

    def init_node_quantities(self, n_node):
        pass

    def init_quadrature_point_quantities(self, n_quad_point):
        pass
