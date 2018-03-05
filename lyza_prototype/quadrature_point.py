from lyza_prototype.helper import determinant, inverse

class QuadraturePoint:
    def __init__(self, coor, weight, label=None):
        self.coor = coor
        self.label = label
        self.weight = weight

    def set_jacobian(self, jac):
        self.jac = jac
        self.det_jac = determinant(jac)
        self.jac_inv_tra = inverse(jac).transpose()

    def set_shape_function(self, N, B):
        self.N = N
        self.B = B

    def set_global_coor(self, val):
        self.global_coor = val

        # import ipdb; ipdb.set_trace()



