import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.assembly_function import AssemblyFunction
from lyza_prototype.matrix_interface import MatrixInterfaceWrapper
from lyza_prototype.vector_interface import VectorInterfaceWrapper
from lyza_prototype.scalar_interface import ScalarInterfaceWrapper

class BilinearForm:
    def __init__(self,
                 function_space_1,
                 function_space_2,
                 # matrix_interface,
                 # quadrature_degree,
                 domain=None):

        # self.matrix_interface = matrix_interface
        self.domain = domain
        # self.quadrature_degree = quadrature_degree

        if function_space_1.mesh != function_space_2.mesh:
            raise Exception('Function spaces not defined on the same mesh')

        self.function_space_1 = function_space_1 # u
        self.function_space_2 = function_space_2 # v

        self.mesh = function_space_1.mesh
        # A^IJ = a(N^J,N^I)


        self.matrix_interface_wrappers = None


    def set_matrix_interface(self, matrix_interface, quadrature_degree):
        self.matrix_interface_wrappers = []

        elems_1 = self.function_space_1.get_finite_elements(quadrature_degree, domain=self.domain)
        elems_2 = self.function_space_2.get_finite_elements(quadrature_degree, domain=self.domain)

        for elem1, elem2 in zip(elems_1, elems_2):
            self.matrix_interface_wrappers.append(MatrixInterfaceWrapper(matrix_interface, elem1, elem2))


    def assemble(self):
        if not self.matrix_interface_wrappers:
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
        bar = progressbar.ProgressBar(max_value=len(self.matrix_interface_wrappers))

        for n, w in enumerate(self.matrix_interface_wrappers):
            bar.update(n+1)
            # matrix =
            elem_matrices.append(w.calculate())

        for w, K_elem in zip(self.matrix_interface_wrappers, elem_matrices):
            for i, I in enumerate(w.elem1.dofmap):
                for j, J in enumerate(w.elem2.dofmap):
                    K[I, J] += K_elem[i,j]

        return K


class LinearForm:
    def __init__(self, function_space, domain=None):
        self.function_space = function_space
        # self.vector_interface = vector_interface
        self.domain = domain
        # self.quadrature_degree = quadrature_degree


    def set_vector_interface(self, vector_interface, quadrature_degree):

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=self.domain)
        self.vector_interface_wrappers = []

        for elem in elems:
            self.vector_interface_wrappers.append(VectorInterfaceWrapper(vector_interface, elem))

    def set_scalar_interface(self, scalar_interface, quadrature_degree):

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=self.domain)
        self.scalar_interface_wrappers = []

        for elem in elems:
            self.scalar_interface_wrappers.append(ScalarInterfaceWrapper(scalar_interface, elem))


    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        logging.info('Calculating element vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(self.vector_interface_wrappers))

        for n, w in enumerate(self.vector_interface_wrappers):
            bar.update(n+1)
            elem_vectors.append(w.calculate())

        for w, f_elem in zip(self.vector_interface_wrappers, elem_vectors):
            for i, I in enumerate(w.elem.dofmap):
                f[I] += f_elem[i]

        return f

    def calculate(self):
        result = 0.

        # assembly = function.function_space.get_assembly(quadrature_degree)
        # interface = AbsoluteErrorScalarInterface(function, exact, p)

        for w in self.scalar_interface_wrappers:
            result += w.calculate()

        return result



