from lyza_prototype.element_scalar import ElementScalar
import numpy as np
import itertools


class AbsoluteErrorElementScalar(ElementScalar):
    def __init__(self, function, exact, p):
        self.function = function
        self.exact = exact
        self.p = p # Lp error

    def calculate(self, elem):
        result = 0.
        n_node = len(elem.nodes)

        coefficients = [self.function.vector[i,0] for i in elem.dofmap]

        for q in elem.quad_points:
            u_h = [0. for i in range(elem.function_dimension)]

            for I, i in itertools.product(range(n_node), range(elem.function_dimension)):
                u_h[i] += q.N[I]*coefficients[I*elem.function_dimension+i]

            exact_val = self.exact(q.global_coor)

            inner_product = 0.
            for i in range(elem.function_dimension):
                inner_product += (exact_val[i] - u_h[i])**2

            result += pow(inner_product, self.p/2.)*q.weight*q.det_jac

        return result


class DerivativeAbsoluteErrorElementScalar(ElementScalar):
    def __init__(self, function, exact_deriv, p):
        self.function = function
        self.exact_deriv = exact_deriv
        self.p = p # Lp error

    def calculate(self, elem):
        result = 0.
        n_node = len(elem.nodes)

        coefficients = [self.function.vector[i,0] for i in elem.dofmap]

        for q in elem.quad_points:
            u_h = np.zeros((elem.function_dimension, elem.physical_dimension))

            for I, i, j in itertools.product(range(n_node), range(elem.function_dimension), range(elem.physical_dimension)):
                u_h[i][j] += q.B[I][j]*coefficients[I*elem.function_dimension+i]

            exact_val = np.array(self.exact_deriv(q.global_coor))

            inner_product = 0.
            for i in range(elem.function_dimension):
                for j in range(elem.physical_dimension):
                    inner_product += (exact_val[i,j] - u_h[i,j])**2


            result += pow(inner_product, self.p/2.)*q.weight*q.det_jac

        return result

