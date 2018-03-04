from lyza_prototype.function_space import FunctionSpace
from lyza_prototype.function import Function
import numpy as np

class NodalProjection:
    def __init__(self, form, function):
        self.form = form
        self.function_size = 6
        self.function = function

        self.function_space = FunctionSpace(
            function.function_space.mesh,
            self.function_size,
            function.function_space.spatial_dimension,
            function.function_space.element_degree)

    def calculate_stresses(self):
        for i in self.form.interfaces:
            i.calculate_stress(self.function)


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

        for q, stress_voigt in zip(interface.elem1.quad_points, interface.stress.vectors):
            for i in range(6):
                f[i] += stress_voigt[i]*q.N[node_idx]*q.det_jac*q.weight

        return f

    def weight_vector(self, interface, node_idx):
        n_dof = self.function_size
        f = np.zeros((n_dof,1))

        for q, stress_voigt in zip(interface.elem1.quad_points, interface.stress.vectors):
            for i in range(6):
                f[i] += q.N[node_idx]*q.det_jac*q.weight

        return f

