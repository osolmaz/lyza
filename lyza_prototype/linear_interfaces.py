from lyza_prototype.element_interface import LinearElementInterface
import numpy as np
import itertools

class FunctionElementVector(LinearElementInterface):

    def __init__(self, function):
        self.function = function

    def vector(self):
        n_node = len(self.elem.nodes)
        n_dof = n_node*self.elem.function_size

        f = np.zeros((n_dof,1))

        for q in self.elem.quad_points:
            f_cont = np.zeros((n_dof,1))

            for I, i in itertools.product(range(n_node), range(self.elem.function_size)):
                alpha = I*self.elem.function_size + i
                f_val = self.function(q.global_coor)
                f[alpha] += f_val[i]*q.N[I]*q.det_jac*q.weight


        return f

class PointLoad(LinearElementInterface):

    def __init__(self, position_function, value):
        self.position_function = position_function
        self.value = value
        self.applied = False
        # self.function = function

    def vector(self):
        n_node = len(self.elem.nodes)
        n_dof = n_node*self.elem.function_size

        f = np.zeros((n_dof,1))

        # for q in self.elem.quad_points:
        #     f_cont = np.zeros((n_dof,1))

        for I in range(n_node):
            # f_val = self.function(q.global_coor)
            if self.position_function(self.elem.nodes[I].coor) and not self.applied:
                for i in range(self.elem.function_size):

                    alpha = I*self.elem.function_size + i
                    f[alpha] += self.value[i]

                self.applied = True
                break


        return f

