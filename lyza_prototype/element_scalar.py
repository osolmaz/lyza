
class ElementScalar:
    def __init__(self):
        pass

    def calculate(self, elem):
        pass

class ElementScalarWrapper:
    def __init__(self, element_scalar, elem):
        self.element_scalar = element_scalar
        self.elem = elem

    def calculate(self):
        return self.element_scalar.calculate(self.elem)
