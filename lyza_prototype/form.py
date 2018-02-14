import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.assembly_function import AssemblyFunction

class BilinearForm:
    def __init__(self,
                 function_space_1,
                 function_space_2,
                 matrix_interface,
                 quadrature_degree,
                 domain=None):

        self.matrix_interface = matrix_interface
        self.domain = domain
        self.quadrature_degree = quadrature_degree

        if function_space_1.mesh != function_space_2.mesh:
            raise Exception('Function spaces not defined on the same mesh')

        self.function_space_1 = function_space_1 # u
        self.function_space_2 = function_space_2 # v

        self.mesh = function_space_1.mesh
        # A^IJ = a(N^J,N^I)

    def assemble(self):
        n_dof_1 = self.function_space_1.get_system_size()
        n_dof_2 = self.function_space_2.get_system_size()
        K = np.zeros((n_dof_2,n_dof_1))
        # K = coo_matrix((n_dof,n_dof))

        assembly_1 = self.function_space_1.get_assembly(self.quadrature_degree, domain=self.domain)
        assembly_2 = self.function_space_2.get_assembly(self.quadrature_degree, domain=self.domain)

        # jac = AssemblyFunction(assembly_1, 1)

        # for e in assembly_1.elems:
            # for points in e.quad_points:
            # print(jac.cell_quad_point_indices[e.parent_cell.idx])
            # val = e.jacobian(
            # quad_points =
            # val =
        # import ipdb; ipdb.set_trace()



        elem_pairs = [[i,j] for i, j in zip(assembly_1.elems, assembly_2.elems)]

        logging.info('Calculating element matrices')
        elem_matrices = []
        bar = progressbar.ProgressBar(max_value=len(elem_pairs))

        for n, e in enumerate(elem_pairs):
            bar.update(n+1)
            matrix = self.matrix_interface.calculate(e[0], e[1])
            elem_matrices.append(matrix)
            # elem_matrices.append(e.calc_matrix(self.element_matrix))

        for e, K_elem in zip(elem_pairs, elem_matrices):
            for i, I in enumerate(e[1].dofmap):
                for j, J in enumerate(e[0].dofmap):
                    K[I, J] += K_elem[i,j]

        return K

    # def calc_element_matrix(self, element1, element2):
    #     pass

class LinearForm:
    def __init__(self, function_space, vector_interface, quadrature_degree, domain=None):
        self.function_space = function_space
        self.vector_interface = vector_interface
        self.domain = domain
        self.quadrature_degree = quadrature_degree
        # self.function_space = function_space

    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        assembly = self.function_space.get_assembly(self.quadrature_degree, domain=self.domain)


        logging.info('Calculating element vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(assembly.elems))

        for n, e in enumerate(assembly.elems):
            bar.update(n+1)
            vector = self.vector_interface.calculate(e)
            elem_vectors.append(vector)
            # elem_matrices.append(e.calc_matrix(self.element_matrix))

        for e, f_elem in zip(assembly.elems, elem_vectors):
            for i, I in enumerate(e.dofmap):
                f[I] += f_elem[i]

        return f



