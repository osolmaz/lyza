
class MatrixInterface:
    def __init__(self):
        pass

    def calculate(self, elem1, elem2):
        pass


class MatrixInterfaceWrapper:
    def __init__(self, matrix_interface, elem1, elem2):
        self.matrix_interface = matrix_interface
        self.elem1 = elem1
        self.elem2 = elem2


    def calculate(self):
        return self.matrix_interface.calculate(self.elem1, self.elem2)
