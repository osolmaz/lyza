from lyza_prototype.cell import Cell
from lyza_prototype.elements import QuadElement1, LineElement1

class Quad(Cell):

    def get_finite_element(self, function_space):
        if function_space.element_degree == 1:
            return QuadElement1(self.nodes, function_space, label=self.label)
        else:
            raise Exception('Invalid element degree: %d'%elem_degree)


class Line(Cell):
    def get_finite_element(self, function_space):
        if function_space.element_degree == 1:
            return LineElement1(self.nodes, function_space, label=self.label)
        else:
            raise Exception('Invalid element degree: %d'%elem_degree)
