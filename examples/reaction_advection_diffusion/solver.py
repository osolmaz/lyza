from lyza_prototype.analytic_solution import get_analytic_solution_vector
from lyza_prototype.solver import apply_bcs, solve_scipy_sparse
from lyza_prototype.function import Function
import logging
import numpy as np

def implicit_euler(m_form, a_form, b_form, u, dirichlet_bcs, u0_function, t_array):

    A = a_form.assemble()
    M = m_form.assemble()

    u0 = get_analytic_solution_vector(u.function_space, u0_function, time=t_array[0])

    # t_array = np.linspace(0, T_MAX, T_RESOLUTION)

    solution_vector = u0
    previous_solution_vector = None

    for i in range(1, len(t_array)):
        # T_GLOBAL = t
        t = t_array[i]
        delta_t = t_array[i] - t_array[i-1]

        b_form.set_time(t)
        b = b_form.assemble()
        matrix = M + delta_t*A
        vector = M.dot(solution_vector) + delta_t*b

        matrix_bc, vector_bc = apply_bcs(matrix, vector, u.function_space, dirichlet_bcs)
        previous_solution_vector = solution_vector
        solution_vector = solve_scipy_sparse(matrix_bc, vector_bc)

        logging.info('T = %f'%(t))

    u.set_vector(solution_vector)
    f = Function(u.function_space)
    f.set_vector(1./delta_t*M.dot(solution_vector-previous_solution_vector)
                 + A.dot(solution_vector))

    return u, f
