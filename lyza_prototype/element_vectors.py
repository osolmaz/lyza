from lyza_prototype.element_interface import LinearElementInterface
import numpy as np
import itertools

class FunctionElementVector(LinearElementInterface):

    def __init__(self, function):
        self.function = function

    def vector(self):
        n_node = len(self.elem.nodes)
        n_dof = n_node*self.elem.function_dimension

        f = np.zeros((n_dof,1))

        for q in self.elem.quad_points:
            f_cont = np.zeros((n_dof,1))

            for I, i in itertools.product(range(n_node), range(self.elem.function_dimension)):
                alpha = I*self.elem.function_dimension + i
                f_val = self.function(q.global_coor)
                f[alpha] += f_val[i]*q.N[I]*q.det_jac*q.weight


        return f
