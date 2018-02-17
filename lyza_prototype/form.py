import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.assembly_function import AssemblyFunction
from lyza_prototype.element_interface import BilinearElementInterface, LinearElementInterface

from copy import deepcopy, copy
import time

class BilinearForm:
    def __init__(
            self,
            function_space_1,
            function_space_2,
            element_interface,
            quadrature_degree,
            domain=None):

        if function_space_1.mesh != function_space_2.mesh:
            raise Exception('Function spaces not defined on the same mesh')

        self.function_space_1 = function_space_1 # u
        self.function_space_2 = function_space_2 # v

        self.mesh = function_space_1.mesh
        # A^IJ = a(N^J,N^I)

        self.interfaces = []

        logging.debug('Getting bilinear form finite elements')
        elems_1 = self.function_space_1.get_finite_elements(quadrature_degree, domain=domain)
        elems_2 = self.function_space_2.get_finite_elements(quadrature_degree, domain=domain)
        logging.debug('Finished getting finite elements')

        for elem1, elem2 in zip(elems_1, elems_2):
            # new_interface = deepcopy(element_interface)
            new_interface = copy(element_interface)
            new_interface.set_elements(elem1, elem2)
            self.interfaces.append(new_interface)

    def assemble(self):

        n_dof_1 = self.function_space_1.get_system_size()
        n_dof_2 = self.function_space_2.get_system_size()
        K = np.zeros((n_dof_2,n_dof_1))
        # K = coo_matrix((n_dof,n_dof))


        logging.info('Calculating element matrices')
        # bar = progressbar.ProgressBar(max_value=len(self.interfaces))

        for n, interface in enumerate(self.interfaces):
            # bar.update(n+1)
            K_elem = interface.matrix()
            for i, I in enumerate(interface.elem1.dofmap):
                for j, J in enumerate(interface.elem2.dofmap):
                    K[I, J] += K_elem[i,j]

        return K

    def __add__(self, a):
        if isinstance(a, BilinearForm):
            return AggregateBilinearForm([self, a])
        elif isinstance(a, AggregateBilinearForm):
            return AggregateBilinearForm([a]+a.bilinear_forms)
        else:
            raise Exception('Cannot add types')

class LinearForm:
    def __init__(
            self,
            function_space,
            element_interface,
            quadrature_degree,
            domain=None):

        self.function_space = function_space

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=domain)
        self.interfaces = []

        for elem in elems:
            # new_interface = deepcopy(element_interface)
            new_interface = copy(element_interface)
            new_interface.set_element(elem)
            self.interfaces.append(new_interface)


    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        logging.info('Calculating element vectors')
        # bar = progressbar.ProgressBar(max_value=len(self.interfaces))

        logging.debug('Getting linear form finite elements')
        for n, interface in enumerate(self.interfaces):
            # bar.update(n+1)
            f_elem = interface.vector()
            for i, I in enumerate(interface.elem.dofmap):
                f[I] += f_elem[i]
        logging.debug('Finished getting finite elements')

        return f

    def calculate(self):
        result = 0.

        for interface in self.interfaces:
            result += interface.evaluate()

        return result

    def __add__(self, a):
        if isinstance(a, LinearForm):
            return AggregateLinearForm([self, a])
        elif isinstance(a, AggregateLinearForm):
            return AggregateLinearForm([a]+a.linear_forms)
        else:
            raise Exception('Cannot add types')


class AggregateBilinearForm:
    def __init__(self, bilinear_forms):
        self.bilinear_forms = bilinear_forms

    def assemble(self):
        return sum([i.assemble() for i in self.bilinear_forms])

    def __add__(self, a):
        if isinstance(a, BilinearForm):
            return AggregateBilinearForm([self.bilinear_forms, a])
        elif isinstance(a, AggregateBilinearForm):
            return AggregateBilinearForm(self.bilinear_forms+a.bilinear_forms)
        else:
            raise Exception('Cannot add types')


    def check(self):
        pass
        # TODO: check vector spaces same

class AggregateLinearForm:
    def __init__(self, linear_forms):
        self.linear_forms = linear_forms

    def assemble(self):
        return sum([i.assemble() for i in self.linear_forms])

    def __add__(self, a):
        if isinstance(a, LinearForm):
            return AggregateLinearForm([self.linear_forms, a])
        elif isinstance(a, AggregateLinearForm):
            return AggregateLinearForm(self.linear_forms+a.linear_forms)
        else:
            raise Exception('Cannot add types')

    def check(self):
        pass
        # TODO: check vector spaces same
