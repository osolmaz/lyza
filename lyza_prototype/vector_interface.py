
class VectorInterface:
    def __init__(self):
        pass

    def calculate(self, elem):
        pass


class VectorInterfaceWrapper:
    def __init__(self, vector_interface, elem):
        self.vector_interface = vector_interface
        self.elem = elem

    def calculate(self):
        return self.vector_interface.calculate(self.elem)

