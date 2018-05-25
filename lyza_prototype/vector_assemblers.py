import itertools
from lyza_prototype.assembler import VectorAssembler
import numpy as np

class FunctionVector(VectorAssembler):
    def set_param(self, function, time):
        self.function = function
        self.time = time

    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size
        K = np.zeros((n_dof, n_dof))

        W_arr = self.mesh.quantities['W'].get_quantity(cell)
        N_arr = self.mesh.quantities['N'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)
        DETJ_arr = self.mesh.quantities['DETJ'].get_quantity(cell)
        XG_arr = self.mesh.quantities['XG'].get_quantity(cell)

        f = np.zeros((n_dof,1))

        for idx in range(len(W_arr)):
            # f_val = self.function(XG_arr[idx], self.time)
            f_val = self.function(XG_arr[idx][:,0].tolist(), self.time)
            N = N_arr[idx]
            W = W_arr[idx][0,0]
            DETJ = DETJ_arr[idx][0,0]

            f_contrib = np.einsum('i,j->j', f_val, N[:,0])*DETJ*W
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
        n_dof = n_node*self.function_size

        f = np.zeros((n_dof,1))

        for I in range(n_node):
            # if self.position_function(self.elements[0].nodes[I].coor) and not self.applied:
            if self.position_function(cell.nodes[I].coor, 0):
                for i in range(self.function_size):

                    alpha = I*self.function_size + i
                    f[alpha] += self.value[i]

                # self.applied = True
                # break


        return f

class ZeroVector(VectorAssembler):
    def calculate_element_vector(self, cell):
        n_node = len(cell.nodes)
        n_dof = n_node*self.function_size

        f = np.zeros((n_dof,1))

        return f

