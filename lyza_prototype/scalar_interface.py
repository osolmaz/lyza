
class ScalarInterface:
    def __init__(self):
        pass

    def calculate(self, elem):
        pass

class ScalarInterfaceWrapper:
    def __init__(self, scalar_interface, elem):
        self.scalar_interface = scalar_interface
        self.elem = elem

    def calculate(self):
        return self.scalar_interface.calculate(self.elem)
