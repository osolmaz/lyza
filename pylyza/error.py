import numpy as np

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
