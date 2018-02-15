
class ElementVector:
    def __init__(self):
        pass

    def calculate(self, elem):
        pass


class ElementVectorWrapper:
    def __init__(self, element_vector, elem):
        self.element_vector = element_vector
        self.elem = elem

    def calculate(self):
        return self.element_vector.calculate(self.elem)

