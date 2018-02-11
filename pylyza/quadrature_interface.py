
class ElementMatrix():
    def __init__(self, param):
        self.param = param
        self.postinit()

    def eval(self, K, N_p, B_p, jac, quad_point, physical_dim, elem_dim, n_dof, n_node):
        pass

    def postinit(self):
        pass

class ElementVector():
    def __init__(self, param):
        self.param = param
        self.postinit()

    def eval(self, K, N_p, B_p, jac, quad_point, physical_dim, elem_dim, n_dof, n_node):
        pass

    def postinit(self):
        pass

