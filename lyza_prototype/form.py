import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.assembly_function import AssemblyFunction
from lyza_prototype.element_matrix import ElementMatrixWrapper
from lyza_prototype.element_vector import ElementVectorWrapper
from lyza_prototype.element_scalar import ElementScalarWrapper

class BilinearForm:
    def __init__(self,
                 function_space_1,
                 function_space_2,
                 # element_matrix,
                 # quadrature_degree,
                 domain=None):

        # self.element_matrix = element_matrix
        self.domain = domain
        # self.quadrature_degree = quadrature_degree

        if function_space_1.mesh != function_space_2.mesh:
            raise Exception('Function spaces not defined on the same mesh')

        self.function_space_1 = function_space_1 # u
        self.function_space_2 = function_space_2 # v

        self.mesh = function_space_1.mesh
        # A^IJ = a(N^J,N^I)


        self.element_matrix_wrappers = None


    def set_element_matrix(self, element_matrix, quadrature_degree):
        self.element_matrix_wrappers = []

        elems_1 = self.function_space_1.get_finite_elements(quadrature_degree, domain=self.domain)
        elems_2 = self.function_space_2.get_finite_elements(quadrature_degree, domain=self.domain)

        for elem1, elem2 in zip(elems_1, elems_2):
            self.element_matrix_wrappers.append(ElementMatrixWrapper(element_matrix, elem1, elem2))


    def assemble(self):
        if not self.element_matrix_wrappers:
            raise Exception('No element matrix assigned to bilinear form')


        n_dof_1 = self.function_space_1.get_system_size()
        n_dof_2 = self.function_space_2.get_system_size()
        K = np.zeros((n_dof_2,n_dof_1))
        # K = coo_matrix((n_dof,n_dof))

        # assembly_1 = self.function_space_1.get_assembly(self.quadrature_degree, domain=self.domain)
        # assembly_2 = self.function_space_2.get_assembly(self.quadrature_degree, domain=self.domain)

        # elem_pairs = [[i,j] for i, j in zip(assembly_1.elems, assembly_2.elems)]

        logging.info('Calculating element matrices')
        elem_matrices = []
        bar = progressbar.ProgressBar(max_value=len(self.element_matrix_wrappers))

        for n, w in enumerate(self.element_matrix_wrappers):
            bar.update(n+1)
            # matrix =
            elem_matrices.append(w.calculate())

        for w, K_elem in zip(self.element_matrix_wrappers, elem_matrices):
            for i, I in enumerate(w.elem1.dofmap):
                for j, J in enumerate(w.elem2.dofmap):
                    K[I, J] += K_elem[i,j]

        return K


class LinearForm:
    def __init__(self, function_space, domain=None):
        self.function_space = function_space
        # self.element_vector = element_vector
        self.domain = domain
        # self.quadrature_degree = quadrature_degree


    def set_element_vector(self, element_vector, quadrature_degree):

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=self.domain)
        self.element_vector_wrappers = []

        for elem in elems:
            self.element_vector_wrappers.append(ElementVectorWrapper(element_vector, elem))

    def set_element_scalar(self, element_scalar, quadrature_degree):

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=self.domain)
        self.element_scalar_wrappers = []

        for elem in elems:
            self.element_scalar_wrappers.append(ElementScalarWrapper(element_scalar, elem))


    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        logging.info('Calculating element vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(self.element_vector_wrappers))

        for n, w in enumerate(self.element_vector_wrappers):
            bar.update(n+1)
            elem_vectors.append(w.calculate())

        for w, f_elem in zip(self.element_vector_wrappers, elem_vectors):
            for i, I in enumerate(w.elem.dofmap):
                f[I] += f_elem[i]

        return f

    def calculate(self):
        result = 0.

        # assembly = function.function_space.get_assembly(quadrature_degree)
        # interface = AbsoluteErrorElementScalar(function, exact, p)

        for w in self.element_scalar_wrappers:
            result += w.calculate()

        return result



