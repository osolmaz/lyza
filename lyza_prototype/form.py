import numpy as np
# from scipy.sparse import coo_matrix
import logging
import progressbar
from lyza_prototype.element_interface import ElementInterface
from lyza_prototype.function_space import FunctionSpace
from lyza_prototype.function import Function

from copy import deepcopy, copy
import time

class Form:
    def project_to_nodes(self, quantity_map, function_space=1):
        if function_space == 1:
            target_space = self.function_space_1
        elif function_space == 2:
            target_space = self.function_space_2
        else:
            raise Exception('Function space can be either 1 or 2')

        quantity_size = quantity_map(self.interfaces[0]).shape[0]

        new_space = FunctionSpace(
            target_space.mesh,
            quantity_size,
            target_space.spatial_dimension,
            target_space.element_degree)

        result = Function(new_space)

        n_dof = new_space.get_system_size()
        f = np.zeros((n_dof,1))
        w = np.zeros((n_dof,1))

        for n, interface in enumerate(self.interfaces):
            target_elem = interface.elements[function_space-1]

            target_quantity = quantity_map(interface)

            for node_i, node in enumerate(target_elem.nodes):
                f_elem = self._projection_vector(target_elem, target_quantity, node_i)
                w_elem = self._projection_weight_vector(target_elem, target_quantity, node_i)
                dofs = new_space.node_dofs[node.idx]

                for dof_i, dof in enumerate(dofs):
                    f[dof] += f_elem[dof_i]
                    w[dof] += w_elem[dof_i]

        projected_values = f/w
        result.set_vector(projected_values)

        return result

    def _projection_vector(self, target_elem, target_quantity, node_idx):
        n_dof = target_quantity.shape[0]
        f = np.zeros((n_dof,1))

        for q, vector in zip(target_elem.quad_points, target_quantity.vectors):
            for i in range(n_dof):
                f[i] += vector[i]*q.N[node_idx]*q.det_jac*q.weight

        return f

    def _projection_weight_vector(self, target_elem, target_quantity, node_idx):
        n_dof = target_quantity.shape[0]
        f = np.zeros((n_dof,1))

        for q in target_elem.quad_points:
            for i in range(n_dof):
                f[i] += q.N[node_idx]*q.det_jac*q.weight

        return f

    def project_to_quadrature_points(self, function, quantity_map, function_space=1):

        for interface in self.interfaces:
            quantity = quantity_map(interface)
            elem = interface.elements[function_space-1]

            for i in range(interface.n_quad_point):
                quantity.vectors[i] = elem.interpolate_at_quad_point(function, i)

    def project_gradient_to_quadrature_points(self, function, quantity_map, function_space=1):

        for interface in self.interfaces:
            quantity = quantity_map(interface)
            elem = interface.elements[function_space-1]

            for i in range(interface.n_quad_point):
                quantity.vectors[i] = elem.interpolate_gradient_at_quad_point(function, i)

    def copy_quantity(self, form, source_quantity_map, target_quantity_map):
        for source_interface in self.interfaces:
            parent_cell = source_interface.elements[0].parent_cell
            target_interface = form.cell_interface_map[parent_cell]
            quantity = quantity_map(interface)


class BilinearForm(Form):
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
        self.cell_interface_map = {}

        logging.debug('Getting bilinear form finite elements')
        elems_1 = self.function_space_1.get_finite_elements(quadrature_degree, domain=domain)
        elems_2 = self.function_space_2.get_finite_elements(quadrature_degree, domain=domain)
        logging.debug('Finished getting finite elements')

        for elem1, elem2 in zip(elems_1, elems_2):
            # new_interface = deepcopy(element_interface)
            new_interface = copy(element_interface)
            new_interface.init_node_quantities(elem1.n_node)
            new_interface.init_quadrature_point_quantities(elem1.n_node)
            new_interface.set_elements([elem1, elem2])
            self.interfaces.append(new_interface)
            self.cell_interface_map[elem1.parent_cell] = new_interface

    def assemble(self):

        n_dof_1 = self.function_space_1.get_system_size()
        n_dof_2 = self.function_space_2.get_system_size()
        K = np.zeros((n_dof_2,n_dof_1))
        # K = coo_matrix((n_dof,n_dof))

        logging.debug('Calculating element matrices')

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            bar = progressbar.ProgressBar(max_value=len(self.interfaces))

        for n, interface in enumerate(self.interfaces):
            K_elem = interface.matrix()
            for i, I in enumerate(interface.elements[0].dofmap):
                for j, J in enumerate(interface.elements[1].dofmap):
                    K[I, J] += K_elem[i,j]

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                bar.update(n+1)
            # print(n)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            bar.finish()

        return K

    def set_time(self, t):
        for i in self.interfaces:
            i.set_time(t)

    def __add__(self, a):
        if isinstance(a, BilinearForm):
            return AggregateBilinearForm([self, a])
        elif isinstance(a, AggregateBilinearForm):
            return AggregateBilinearForm([self]+a.bilinear_forms)
        else:
            raise Exception('Cannot add types')



class LinearForm(Form):
    def __init__(
            self,
            function_space,
            element_interface,
            quadrature_degree,
            domain=None):

        self.function_space = function_space

        elems = self.function_space.get_finite_elements(quadrature_degree, domain=domain)
        self.interfaces = []
        self.cell_interface_map = {}

        for elem in elems:
            # new_interface = deepcopy(element_interface)
            new_interface = copy(element_interface)
            new_interface.init_quadrature_point_quantities(elem.n_node)
            new_interface.init_node_quantities(elem.n_node)
            new_interface.set_elements([elem])
            self.interfaces.append(new_interface)
            self.cell_interface_map[elem.parent_cell] = new_interface


    def assemble(self):
        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))

        logging.debug('Calculating element vectors')
        # bar = progressbar.ProgressBar(max_value=len(self.interfaces))

        logging.debug('Getting linear form finite elements')
        for n, interface in enumerate(self.interfaces):
            # bar.update(n+1)
            f_elem = interface.vector()
            for i, I in enumerate(interface.elements[0].dofmap):
                f[I] += f_elem[i]
        logging.debug('Finished getting finite elements')

        return f

    def calculate(self):
        result = 0.

        for interface in self.interfaces:
            result += interface.evaluate()

        return result

    def set_time(self, t):
        for i in self.interfaces:
            i.set_time(t)

    def __add__(self, a):
        if isinstance(a, LinearForm):
            return AggregateLinearForm([self, a])
        elif isinstance(a, AggregateLinearForm):
            return AggregateLinearForm([self]+a.linear_forms)
        else:
            raise Exception('Cannot add types')


class AggregateBilinearForm:
    def __init__(self, bilinear_forms):
        self.bilinear_forms = bilinear_forms

    def assemble(self):
        return sum([i.assemble() for i in self.bilinear_forms])

    def __add__(self, a):
        if isinstance(a, BilinearForm):
            return AggregateBilinearForm(self.bilinear_forms+[a])
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
            return AggregateLinearForm(self.linear_forms+[a])
        elif isinstance(a, AggregateLinearForm):
            return AggregateLinearForm(self.linear_forms+a.linear_forms)
        else:
            raise Exception('Cannot add types')

    def check(self):
        pass
        # TODO: check vector spaces same
