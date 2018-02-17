

class BilinearElementInterface:

    def __init__(self):
        pass

    def matrix(self):
        pass

    def evaluate(self):
        pass

    def set_elements(self, elem1, elem2):
        self.elem1 = elem1
        self.elem2 = elem2

class LinearElementInterface:

    def __init__(self):
        pass

    def vector(self):
        pass

    def evaluate(self):
        pass

    def set_element(self, elem):
        self.elem = elem


# class BilinearElementInterfaceWrapper:
#     def __init__(self, element_interface, elem1, elem2):
#         self.element_interface = element_interface
#         self.elem1 = elem1
#         self.elem2 = elem2

#     def bilinear_form_matrix(self):
#         return self.element_interface.bilinear_form_matrix(self.elem1, self.elem2)

#     def evaluate_bilinear_form(self):
#         return self.element_interface.evaluate_bilinear_form(self.elem1, self.elem2)


# class LinearElementInterfaceWrapper:
#     def __init__(self, element_interface, elem):
#         self.element_interface = element_interface
#         self.elem = elem

#     def linear_form_vector(self):
#         return self.element_interface.linear_form_vector(self.elem)

#     def evaluate_linear_form(self):
#         return self.element_interface.evaluate_linear_form(self.elem)


