import itertools
from lyza.assembler import VectorAssembler
import numpy as np


class FunctionVector(VectorAssembler):
    def set_param(self, function, time):
        self.function = function
        self.time = time

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        N_arr = self.mesh.quantities["N"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)
        XG_arr = self.mesh.quantities["XG"].get_quantity(cell)

        f = np.zeros((n_dof, 1))

        for idx in range(len(W_arr)):
            # f_val = self.function(XG_arr[idx], self.time)
            f_val = self.function(XG_arr[idx][:, 0].tolist(), self.time)
            N = N_arr[idx][:, 0]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]

            f_contrib = np.einsum("i,j->ji", f_val, N) * DETJ * W

            f_contrib = f_contrib.reshape(f.shape)
            f += f_contrib

            # for I, i in itertools.product(range(n_node), range(self.function_size)):
            #     alpha = I*self.function_size + i
            #     f[alpha] += f_val[i]*N[I,0]*DETJ*W

        return f


class PointLoadVector(VectorAssembler):
    def set_param(self, position_function, value):
        self.position_function = position_function
        self.value = value
        # self.applied = False
        # self.function = function

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        f = np.zeros((n_dof, 1))

        for I in range(n_node):
            # if self.position_function(self.elements[0].nodes[I].coor) and not self.applied:
            if self.position_function(cell.nodes[I].coor, 0):
                for i in range(self.function_size):

                    alpha = I * self.function_size + i
                    f[alpha] += self.value[i]

                # self.applied = True
                # break

        return f


class ZeroVector(VectorAssembler):
    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        f = np.zeros((n_dof, 1))

        return f


class InelasticityResidualVector(VectorAssembler):
    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        f = np.zeros((n_dof, 1))

        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)
        SIG_arr = self.mesh.quantities["SIG"].get_quantity(cell)

        for idx in range(len(W_arr)):
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]
            SIG = SIG_arr[idx]

            f_contrib = -np.einsum("ab, ib -> ia", SIG, B) * DETJ * W
            f_contrib = f_contrib.reshape(f.shape)
            f += f_contrib

        return f


class HyperelasticityResidual(VectorAssembler):
    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node * self.function_size

        FINVT_arr = self.mesh.quantities["FINVT"].get_quantity(cell)
        BBAR_arr = self.mesh.quantities["BBAR"].get_quantity(cell)
        B_arr = self.mesh.quantities["B"].get_quantity(cell)
        LCG_arr = self.mesh.quantities["LCG"].get_quantity(cell)
        TAU_arr = self.mesh.quantities["TAU"].get_quantity(cell)
        W_arr = self.mesh.quantities["W"].get_quantity(cell)
        DETJ_arr = self.mesh.quantities["DETJ"].get_quantity(cell)

        f = np.zeros((n_dof, 1))

        for idx in range(len(W_arr)):
            Finvtra = FINVT_arr[idx]
            Bbar = BBAR_arr[idx]
            B = B_arr[idx]
            W = W_arr[idx][0, 0]
            DETJ = DETJ_arr[idx][0, 0]
            b_ = LCG_arr[idx]
            tau = TAU_arr[idx]

            n_node = B.shape[0]
            spatial_dim = B.shape[1]

            f_higher = -np.einsum("ab, ib -> ia", tau, Bbar) * DETJ * W
            f_higher = f_higher.reshape((n_node * spatial_dim, 1))
            f += f_higher

        return f
