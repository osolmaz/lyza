from lyza_prototype.quadrature_interface import ElementVector
import itertools

class FunctionVector(ElementVector):
    def __init__(self, function):
        self.function = function
        super().__init__()

    def eval(self, f, N_p, B_p, det_jac, quad_point, function_dim, physical_dim, elem_dim, n_dof, n_node):
        for I, i in itertools.product(range(n_node), range(function_dim)):
            alpha = I*function_dim + i
            f_val = self.function(quad_point)
            # print(alpha, i, I, f_val, function_dim)
            f[alpha] += f_val[i]*N_p[I]*det_jac

