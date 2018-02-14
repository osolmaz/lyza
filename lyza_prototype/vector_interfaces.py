from lyza_prototype.vector_interface import VectorInterface
import numpy as np
import itertools

class FunctionVectorInterface(VectorInterface):

    def __init__(self, function):
        self.function = function

    def calculate(self, elem):
        n_node = len(elem.nodes)
        n_dof = n_node*elem.function_dimension

        f = np.zeros((n_dof,1))

        for n in range(elem.n_quad_point):
            f_cont = np.zeros((n_dof,1))

            for I, i in itertools.product(range(n_node), range(elem.function_dimension)):
                alpha = I*elem.function_dimension + i
                f_val = self.function(elem.quad_points_global[n])
                # print(alpha, i, I, f_val, function_dim)
                f[alpha] += f_val[i]*elem.quad_N[n][I]*elem.quad_det_jac[n]*elem.quad_points[n].weight


        return f
