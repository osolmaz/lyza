from pylyza.quadrature_interface import ElementVector
import itertools

class FunctionVector(ElementVector):
    def eval(self, f, N_p, B_p, jac, quad_point, physical_dim, elem_dim, n_dof, n_node):
        for I, i in itertools.product(range(n_node), range(physical_dim)):
            alpha = I*physical_dim + i
            f_val = self.function(quad_point)

            f[alpha] += f_val[i]*N_p[I]*jac

    def postinit(self):
        self.function = self.param['function']
