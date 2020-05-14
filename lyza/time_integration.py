from lyza.analytic_solution import get_analytic_solution_vector
from lyza.solver import apply_bcs, solve_scipy_sparse
from lyza.function import Function
from lyza.vtk import VTKFile
import logging
import numpy as np
import progressbar


def time_array(t_init, t_max, delta_t):
    result = [t_init]

    while result[-1] < t_max:
        result.append(result[-1] + delta_t)

    return result


def implicit_euler(
    m_form, a_form, b_form, dirichlet_bcs, u0_function, t_array, out_prefix=None
):

    mesh = a_form.mesh
    function_size = a_form.function_size
    node_dofs = a_form.node_dofs

    A = a_form.assemble()
    M = m_form.assemble()

    u = Function(mesh, function_size)
    u.set_analytic_solution(u0_function)

    solution_vector = u.vector
    previous_solution_vector = None

    bar = progressbar.ProgressBar(max_value=len(t_array))

    if out_prefix:
        # u.set_vector(solution_vector)
        u.set_label("u")
        ofile = VTKFile("%s%05d.vtk" % (out_prefix, 0))
        ofile.write(mesh, u)

    for i in range(1, len(t_array)):
        t = t_array[i]
        delta_t = t_array[i] - t_array[i - 1]

        b_form.set_time(t)
        b = b_form.assemble()
        matrix = M + delta_t * A
        vector = M.dot(solution_vector) + delta_t * b

        for bc in dirichlet_bcs:
            bc.set_time(t)

        matrix_bc, vector_bc = apply_bcs(
            matrix, vector, mesh, node_dofs, function_size, dirichlet_bcs
        )
        # matrix_bc, vector_bc = apply_bcs(matrix, vector, u.function_space, dirichlet_bcs)
        previous_solution_vector = solution_vector
        # solution_vector = solve_petsc(matrix_bc, vector_bc)
        solution_vector = solve_scipy_sparse(matrix_bc, vector_bc)

        if out_prefix:
            u.set_vector(solution_vector)
            ofile = VTKFile("%s%05d.vtk" % (out_prefix, i))
            ofile.write(mesh, u)

        bar.update(i + 1)
        # logging.info('T = %f'%(t))
    bar.finish()

    u.set_vector(solution_vector)
    f = Function(mesh, function_size)
    f.set_vector(
        1.0 / delta_t * M.dot(solution_vector - previous_solution_vector)
        + A.dot(solution_vector)
    )

    return u, f
