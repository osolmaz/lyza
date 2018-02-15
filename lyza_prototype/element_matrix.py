
class ElementMatrix:
    def __init__(self):
        pass

    def calculate(self, elem1, elem2):
        pass


class ElementMatrixWrapper:
    def __init__(self, element_matrix, elem1, elem2):
        self.element_matrix = element_matrix
        self.elem1 = elem1
        self.elem2 = elem2


    def calculate(self):
        return self.element_matrix.calculate(self.elem1, self.elem2)
