from lyza_prototype.element_interface import ElementInterface
import numpy as np
import itertools

# class FunctionInterface(ElementInterface):

#     def __init__(self, function):
#         self.function = function

#     def vector(self):
#         n_node = len(self.elements[0].nodes)
#         n_dof = n_node*self.elements[0].function_size

#         f = np.zeros((n_dof,1))

#         for q in self.elements[0].quad_points:
#             f_cont = np.zeros((n_dof,1))

#             for I, i in itertools.product(range(n_node), range(self.elements[0].function_size)):
#                 alpha = I*self.elements[0].function_size + i
#                 f_val = self.function(q.global_coor)
#                 f[alpha] += f_val[i]*q.N[I]*q.det_jac*q.weight

#         return f

class FunctionInterface(ElementInterface):

    def __init__(self, function):
        self.function = function

    def vector(self):
        n_node = len(self.elements[0].nodes)
        n_dof = n_node*self.elements[0].function_size

        f = np.zeros((n_dof,1))

        for q in self.elements[0].quad_points:

            for I, i in itertools.product(range(n_node), range(self.elements[0].function_size)):
                alpha = I*self.elements[0].function_size + i
                f_val = self.function(q.global_coor, self.time)
                f[alpha] += f_val[i]*q.N[I]*q.det_jac*q.weight

        return f


class PointLoad(ElementInterface):

    def __init__(self, position_function, value):
        self.position_function = position_function
        self.value = value
        # self.applied = False
        # self.function = function

    def vector(self):
        n_node = len(self.elements[0].nodes)
        n_dof = n_node*self.elements[0].function_size

        f = np.zeros((n_dof,1))

        for I in range(n_node):
            # if self.position_function(self.elements[0].nodes[I].coor) and not self.applied:
            if self.position_function(self.elements[0].nodes[I].coor):
                for i in range(self.elements[0].function_size):

                    alpha = I*self.elements[0].function_size + i
                    f[alpha] += self.value[i]

                # self.applied = True
                # break


        return f

class ZeroVector(ElementInterface):
    def vector(self):
        n_node = len(self.elements[0].nodes)
        n_dof = n_node*self.elements[0].function_size

        f = np.zeros((n_dof,1))

        return f
