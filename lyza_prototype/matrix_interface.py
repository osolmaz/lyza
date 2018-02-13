
class MatrixInterface:
    def __init__(self):
        pass

    def calculate(self, elem1, elem2):
        pass
        # n_dof = len(self.nodes)*self.function_dimension
        # K = np.zeros((n_dof,n_dof))
        # n_node = len(self.nodes)

        # for n in range(self.n_quad_point):
        #     K_cont = np.zeros((n_dof,n_dof))
        #     quad_matrix.eval(
        #         K_cont,
        #         self.quad_N[n],
        #         self.quad_B[n],
        #         self.quad_det_jac[n],
        #         self.quad_points_global[n],
        #         self.function_dimension,
        #         self.physical_dimension,
        #         self.elem_dim,
        #         n_dof,
        #         n_node)

        #     K = K + self.quad_weights[n]*K_cont
        # # import ipdb; ipdb.set_trace()
        # return K

