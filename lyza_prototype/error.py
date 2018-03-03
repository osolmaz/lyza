from math import log
import numpy as np
from lyza_prototype.form import LinearForm
from lyza_prototype.element_interface import LinearElementInterface
from lyza_prototype.analytic_solution import get_analytic_solution_vector
import logging
import itertools


class LpNormInterface(LinearElementInterface):
    def __init__(self, function, exact, p):
        self.function = function
        self.exact = exact
        self.p = p # Lp error

    def evaluate(self):
        result = 0.
        n_node = len(self.elem.nodes)

        coefficients = [self.function.vector[i,0] for i in self.elem.dofmap]

        for q in self.elem.quad_points:
            u_h = [0. for i in range(self.elem.function_size)]

            for I, i in itertools.product(range(n_node), range(self.elem.function_size)):
                u_h[i] += q.N[I]*coefficients[I*self.elem.function_size+i]

            exact_val = self.exact(q.global_coor)

            inner_product = 0.
            for i in range(self.elem.function_size):
                inner_product += (exact_val[i] - u_h[i])**2

            result += pow(inner_product, self.p/2.)*q.weight*q.det_jac

        return result


class LinfNormInterface(LinearElementInterface):
    def __init__(self, function, exact):
        self.function = function
        self.exact = exact

        self.quad_point_values = None

    def evaluate(self):
        n_node = len(self.elem.nodes)

        coefficients = [self.function.vector[i,0] for i in self.elem.dofmap]

        self.quad_point_values = []

        for q in self.elem.quad_points:
            u_h = [0. for i in range(self.elem.function_size)]

            for I, i in itertools.product(range(n_node), range(self.elem.function_size)):
                u_h[i] += q.N[I]*coefficients[I*self.elem.function_size+i]

            exact_val = self.exact(q.global_coor)

            inner_product = 0.
            for i in range(self.elem.function_size):
                inner_product += (exact_val[i] - u_h[i])**2

            linf = pow(inner_product, 0.5)
            self.quad_point_values.append(linf)

        return 0


class DerivativeLpNormInterface(LinearElementInterface):
    def __init__(self, function, exact_deriv, p):
        self.function = function
        self.exact_deriv = exact_deriv
        self.p = p # Lp error

    def evaluate(self):
        result = 0.
        n_node = len(self.elem.nodes)

        coefficients = [self.function.vector[i,0] for i in self.elem.dofmap]

        for q in self.elem.quad_points:
            u_h = np.zeros((self.elem.function_size, self.elem.spatial_dimension))

            for I, i, j in itertools.product(range(n_node), range(self.elem.function_size), range(self.elem.spatial_dimension)):
                u_h[i][j] += q.B[I][j]*coefficients[I*self.elem.function_size+i]

            exact_val = np.array(self.exact_deriv(q.global_coor))

            inner_product = 0.
            for i in range(self.elem.function_size):
                for j in range(self.elem.spatial_dimension):
                    inner_product += (exact_val[i,j] - u_h[i,j])**2


            result += pow(inner_product, self.p/2.)*q.weight*q.det_jac

        return result



def absolute_error(function, exact, exact_deriv, quadrature_degree, error='l2', time=0):
    logging.debug('Calculating error')
    if error == 'l2':
        result = absolute_error_lp(function, exact, 2, quadrature_degree, time=time)
    elif error == 'linf':
        # TODO: decide on how to calculate the infinity norm
        # result = abs(function.vector - get_analytic_solution_vector(function.function_space, exact)).max()
        result = absolute_error_linf(function, exact, quadrature_degree, time=time)
    elif error == 'h1':
        l2 = absolute_error_lp(function, exact, 2, quadrature_degree, time=time)
        l2d = absolute_error_deriv_lp(function, exact_deriv, 2, quadrature_degree, time=time)
        result = pow(pow(l2,2.) + pow(l2d,2.), .5)
    else:
        raise Exception('Invalid error specification: %s'%error)
    logging.debug('Done')

    return result


def absolute_error_lp(function, exact, p, quadrature_degree, time=0):
    form = LinearForm(
        function.function_space,
        LpNormInterface(function, lambda x: exact(x, time), p),
        quadrature_degree)

    result = form.calculate()
    result = pow(result, 1./p)
    return result

def absolute_error_linf(function, exact, quadrature_degree, time=0):
    form = LinearForm(
        function.function_space,
        LinfNormInterface(function, lambda x: exact(x, time)),
        quadrature_degree)

    form.calculate()

    all_values = []
    for i in form.interfaces:
        all_values += i.quad_point_values

    result = max(all_values)
    return result


def absolute_error_deriv_lp(function, exact_deriv, p, quadrature_degree, time=0):
    form = LinearForm(
        function.function_space,
        DerivativeLpNormInterface(function, lambda x: exact_deriv(x, time), p),
        quadrature_degree)

    result = form.calculate()
    result = pow(result, 1./p)

    return result


def plot_convergence_rates(path, h_max_array, l2=None, linf=None, h1=None):
    import pylab as pl

    pl.figure()

    if l2:
        pl.semilogx(
            h_max_array,
            calculate_convergence(h_max_array, l2),
            '-o', label='$L^2$ Convergence rate')
    if linf:
        pl.semilogx(
            h_max_array,
            calculate_convergence(h_max_array, linf),
            '-o', label='$L^\infty$ Convergence rate')
    if h1:
        pl.semilogx(
            h_max_array,
            calculate_convergence(h_max_array, h1),
            '-o', label='$H^1$ Convergence rate')

    pl.xlabel('$h_{max}$')
    pl.ylabel('$\log(\epsilon_{n-1}-\epsilon_{n})/\log(h_{max,n-1}-h_{max,n})$')
    pl.grid(b=True, which='minor', color='gray', linestyle='--')
    pl.grid(b=True, which='major', color='gray', linestyle='-')
    pl.title('Convergence rates')
    pl.legend()

    pl.savefig(path)

def calculate_convergence(h_max_array, error_array):
    convergence_array = [float('nan')]

    for i in range(len(h_max_array)):
        if i >= 1:
            base = h_max_array[i-1]/h_max_array[i]
            # import ipdb; ipdb.set_trace()
            convergence_array.append(log(error_array[i-1]/error_array[i])/log(base))

    return convergence_array

def plot_errors(path, h_max_array, l2=None, linf=None, h1=None):
    import pylab as pl

    pl.figure()

    # Error figure
    if l2:
        pl.loglog(h_max_array, l2, '-o', label='$L^2$ Error')
    if linf:
        pl.loglog(h_max_array, linf, '-o', label='$L^\infty$ Error')
    if h1:
        pl.loglog(h_max_array, h1, '-o', label='$H^1$ Error')


    # pl.minorticks_on()
    pl.xlabel('$h_{max}$')
    pl.ylabel('$\epsilon_{a}$')
    pl.grid(b=True, which='minor', color='gray', linestyle='--')
    pl.grid(b=True, which='major', color='gray', linestyle='-')
    pl.title('Errors')
    pl.legend()

    pl.savefig(path)


