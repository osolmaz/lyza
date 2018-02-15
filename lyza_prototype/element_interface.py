

class ElementInterface:

    def __init__(self):
        pass

    def bilinear_form_matrix(self, elem1, elem2):
        pass

    def linear_form_vector(self, elem):
        pass

    def evaluate_bilinear_form(self, elem1, elem2):
        pass

    def evaluate_linear_form(self, elem):
        pass


class BilinearElementInterfaceWrapper:
    def __init__(self, element_interface, elem1, elem2):
        self.element_interface = element_interface
        self.elem1 = elem1
        self.elem2 = elem2

    def bilinear_form_matrix(self):
        return self.element_interface.bilinear_form_matrix(self.elem1, self.elem2)

    def evaluate_bilinear_form(self):
        return self.element_interface.evaluate_bilinear_form(self.elem1, self.elem2)


class LinearElementInterfaceWrapper:
    def __init__(self, element_interface, elem):
        self.element_interface = element_interface
        self.elem = elem

    def linear_form_vector(self):
        return self.element_interface.linear_form_vector(self.elem)

    def evaluate_linear_form(self):
        return self.element_interface.evaluate_linear_form(self.elem)


