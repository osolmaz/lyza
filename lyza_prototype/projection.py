from lyza_prototype.function_space import FunctionSpace
from lyza_prototype.function import Function
import numpy as np

class NodalProjection:
    def __init__(self, form, function, quantity_map):
        self.form = form
        self.quantity_map = quantity_map
        self.function_size = quantity_map(self.form.interfaces[0]).shape[0]
        self.function = function

        self.function_space = FunctionSpace(
            function.function_space.mesh,
            self.function_size,
            function.function_space.spatial_dimension,
            function.function_space.element_degree)

    def calculate(self):

        result = Function(self.function_space)

        n_dof = self.function_space.get_system_size()
        f = np.zeros((n_dof,1))
        w = np.zeros((n_dof,1))

        for n, interface in enumerate(self.form.interfaces):

            for node_i, node in enumerate(interface.elem1.nodes):
                f_elem = self.vector(interface, node_i)
                w_elem = self.weight_vector(interface, node_i)
                dofs = self.function_space.node_dofs[node.idx]

                for dof_i, dof in enumerate(dofs):
                    f[dof] += f_elem[dof_i]
                    w[dof] += w_elem[dof_i]

        projected_values = f/w
        # import ipdb; ipdb.set_trace()
        result.set_vector(projected_values)

        return result

    def vector(self, interface, node_idx):
        n_dof = self.function_size
        f = np.zeros((n_dof,1))

        for q, vector in zip(interface.elem1.quad_points, self.quantity_map(interface).vectors):
            for i in range(vector.shape[0]):
                f[i] += vector[i]*q.N[node_idx]*q.det_jac*q.weight

        return f

    def weight_vector(self, interface, node_idx):
        n_dof = self.function_size
        f = np.zeros((n_dof,1))

        for q in interface.elem1.quad_points:
            for i in range(6):
                f[i] += q.N[node_idx]*q.det_jac*q.weight

        return f

