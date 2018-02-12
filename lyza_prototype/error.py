import pylab as pl
import numpy as np
from math import log

def get_exact_solution_vector(function_space, exact):
    exact_solution_vector = np.zeros((function_space.get_system_size(), 1))

    for n in function_space.mesh.nodes:
        exact_val = exact(n.coor)
        for n, dof in enumerate(function_space.node_dofs[n.idx]):
            exact_solution_vector[dof] = exact_val[n]

    return exact_solution_vector

def absolute_error(function, exact, exact_deriv, error='l2'):

    if error == 'l2':
        result = absolute_error_lp(function, exact, 2)
    elif error == 'linf':
        result = abs(function.vector - get_exact_solution_vector(function.function_space, exact)).max()
    elif error == 'h1':
        l2 = absolute_error_lp(function, exact, 2)
        l2d = absolute_error_deriv_lp(function, exact_deriv, 2)
        result = pow(pow(l2,2.) + pow(l2d,2.), .5)
    else:
        raise Exception('Invalid error specification: %s'%error)

    return result

def absolute_error_lp(function, exact, p):
    result = 0.

    for e in function.function_space.get_finite_elements():
        coefficients = [function.vector[i,0] for i in e.dofmap]
        import ipdb; ipdb.set_trace()
        result += e.absolute_error_lp(exact, coefficients, p)

    result = pow(result, 1./p)

    return result

def absolute_error_deriv_lp(function, exact_deriv, p):
    result = 0.

    for e in function.function_space.get_finite_elements():
        coefficients = [function.vector[i,0] for i in e.dofmap]
        result += e.absolute_error_deriv_lp(exact_deriv, coefficients, p)

    result = pow(result, 1./p)

    return result


def plot_convergence_rates(path, h_max_array, l2_array, linf_array, h1_array):
    pl.figure()

    linf_convergence_array = [float('nan')]
    l2_convergence_array = [float('nan')]
    h1_convergence_array = [float('nan')]

    for i in range(len(h_max_array)):
        if i >= 1:
            denominator = log(h_max_array[i-1]-h_max_array[i])
            l2_convergence_array.append(log(l2_array[i-1]-l2_array[i])/denominator)
            linf_convergence_array.append(log(linf_array[i-1]-linf_array[i])/denominator)
            h1_convergence_array.append(log(h1_array[i-1]-h1_array[i])/denominator)
    # import ipdb; ipdb.set_trace()

    # print(l2_convergence_array)
    pl.semilogx(h_max_array, l2_convergence_array, '-o', label='$L^2$ Convergence rate')
    pl.semilogx(h_max_array, linf_convergence_array, '-o', label='$L^\infty$ Convergence rate')
    pl.semilogx(h_max_array, h1_convergence_array, '-o', label='$H^1$ Convergence rate')

    pl.xlabel('$h_{max}$')
    pl.ylabel('$\log(\epsilon_{n-1}-\epsilon_{n})/\log(h_{max,n-1}-h_{max,n})$')
    pl.grid(b=True, which='minor', color='gray', linestyle='--')
    pl.grid(b=True, which='major', color='gray', linestyle='-')
    pl.title('Convergence rates')
    pl.legend()

    pl.savefig(path)


def plot_errors(path, h_max_array, l2_array, linf_array, h1_array):
    pl.figure()

    # Error figure
    pl.loglog(h_max_array, l2_array, '-o', label='$L^2$ Error')
    pl.loglog(h_max_array, linf_array, '-o', label='$L^\infty$ Error')
    pl.loglog(h_max_array, h1_array, '-o', label='$H^1$ Error')


    # pl.minorticks_on()
    pl.xlabel('$h_{max}$')
    pl.ylabel('$\epsilon_{a}$')
    pl.grid(b=True, which='minor', color='gray', linestyle='--')
    pl.grid(b=True, which='major', color='gray', linestyle='-')
    pl.title('Errors')
    pl.legend()

    pl.savefig(path)


