import numpy as np
import logging
import progressbar

class BilinearForm:
    def __init__(self, element_matrix, domain=None):
        self.element_matrix = element_matrix
        self.domain = domain
        # self.function_space = function_space

    def assemble(self, function_space):
        n_dof = function_space.get_system_size()
        K = np.zeros((n_dof,n_dof))
        # K = csr_matrix((n_dof,n_dof))

        elems = function_space.get_finite_elements()

        logging.info('Calculating element matrices')
        elem_matrices = []
        bar = progressbar.ProgressBar(max_value=len(elems))
        for n, e in enumerate(elems):
            bar.update(n+1)
            elem_matrices.append(e.calc_matrix(self.element_matrix))

        for e, K_elem in zip(elems, elem_matrices):
            for i, I in enumerate(e.dofmap):
                for j, J in enumerate(e.dofmap):
                    K[I, J] += K_elem[i,j]

        return (K+K.T)/2.


class LinearForm:
    def __init__(self, element_vector, domain=None):
        self.element_vector = element_vector
        self.domain = domain
        # self.function_space = function_space

    def assemble(self, function_space):
        n_dof = function_space.get_system_size()
        f = np.zeros((n_dof,1))

        elems = function_space.get_finite_elements()

        logging.info('Calculating element force vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(elems))
        for n, e in enumerate(elems):
            bar.update(n+1)
            elem_vectors.append(e.calc_vector(self.element_vector))

        for e, f_elem in zip(elems, elem_vectors):
            for i, I in enumerate(e.dofmap):
                f[I] += f_elem[i]

        return f

