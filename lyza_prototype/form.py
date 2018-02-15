import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.assembly_function import AssemblyFunction
# from lyza_prototype.element_matrix import ElementMatrixWrapper
# from lyza_prototype.element_vector import ElementVectorWrapper
# from lyza_prototype.element_scalar import ElementScalarWrapper
from lyza_prototype.element_interface import ElementInterface, \
    BilinearElementInterfaceWrapper, LinearElementInterfaceWrapper

class BilinearForm:
    def __init__(self,
                 function_space_1,
                 function_space_2):

        if function_space_1.mesh != function_space_2.mesh:
            raise Exception('Function spaces not defined on the same mesh')

        self.function_space_1 = function_space_1 # u
        self.function_space_2 = function_space_2 # v

        self.mesh = function_space_1.mesh
        # A^IJ = a(N^J,N^I)

        self.wrappers = None


    def set_element_interface(self, element_interface, quadrature_degree, domain=None):
        self.wrappers = []

        elems_1 = self.function_space_1.get_finite_elements(quadrature_degree, domain=domain)
        elems_2 = self.function_space_2.get_finite_elements(quadrature_degree, domain=domain)

        for elem1, elem2 in zip(elems_1, elems_2):
            self.wrappers.append(BilinearElementInterfaceWrapper(element_interface, elem1, elem2))


    def assemble(self):
        if not self.wrappers:
            raise Exception('No element interface assigned to bilinear form')


        n_dof_1 = self.function_space_1.get_system_size()
        n_dof_2 = self.function_space_2.get_system_size()
        K = np.zeros((n_dof_2,n_dof_1))
        # K = coo_matrix((n_dof,n_dof))


        logging.info('Calculating element matrices')
        elem_matrices = []
        bar = progressbar.ProgressBar(max_value=len(self.wrappers))

        for n, w in enumerate(self.wrappers):
            bar.update(n+1)
            elem_matrices.append(w.bilinear_form_matrix())

        for w, K_elem in zip(self.wrappers, elem_matrices):
            for i, I in enumerate(w.elem1.dofmap):
                for j, J in enumerate(w.elem2.dofmap):
                    K[I, J] += K_elem[i,j]

        return K


class LinearForm:
    def __init__(self, function_space):
        self.function_space = function_space


    def set_element_interface(self, element_interface, quadrature_degree, domain=None):

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=domain)
        self.wrappers = []

        for elem in elems:
            self.wrappers.append(LinearElementInterfaceWrapper(element_interface, elem))


    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        logging.info('Calculating element vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(self.wrappers))

        for n, w in enumerate(self.wrappers):
            bar.update(n+1)
            elem_vectors.append(w.linear_form_vector())

        for w, f_elem in zip(self.wrappers, elem_vectors):
            for i, I in enumerate(w.elem.dofmap):
                f[I] += f_elem[i]

        return f

    def calculate(self):
        result = 0.

        for w in self.wrappers:
            result += w.evaluate_linear_form()

        return result



