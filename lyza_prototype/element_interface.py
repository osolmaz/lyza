

class BilinearElementInterface:

    def __init__(self):
        pass

    def matrix(self):
        raise Exception('Undefined element interface')

    def evaluate(self):
        raise Exception('Undefined element interface')

    def set_elements(self, elem1, elem2):
        self.elem1 = elem1
        self.elem2 = elem2
        self.n_quad_point = elem1.n_quad_point

        self.time = 0

    def set_time(self, t):
        self.time = t

    def init_node_quantities(self, n_node):
        pass

    def init_quadrature_point_quantities(self, n_quad_point):
        pass

class LinearElementInterface:

    def __init__(self):
        pass

    def vector(self):
        raise Exception('Undefined element interface')

    def evaluate(self):
        raise Exception('Undefined element interface')

    def set_time(self, t):
        self.time = t

    def set_element(self, elem):
        self.elem = elem
        self.n_quad_point = elem.n_quad_point

        self.time = 0

    def init_node_quantities(self, n_node):
        pass

    def init_quadrature_point_quantities(self, n_quad_point):
        pass
