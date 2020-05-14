import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time

from lyza.function import Function
from lyza.vtk import VTKFile


def solve(
    matrix_assembler,
    vector_assembler,
    dirichlet_bcs,
    solver="scipy_sparse",
    solver_parameters={},
):

    function = Function(matrix_assembler.mesh, matrix_assembler.function_size)

    A = matrix_assembler.assemble()
    f_bc = vector_assembler.assemble()

    start_time = time.time()
    A_bc, f_bc = apply_bcs(
        A,
        f_bc,
        matrix_assembler.mesh,
        matrix_assembler.node_dofs,
        matrix_assembler.function_size,
        dirichlet_bcs,
    )
    logging.debug("Applied bcs in %f sec" % (time.time() - start_time))

    n_dof = A.shape[0]

    logging.debug("Attempting to solve %dx%d system" % (n_dof, n_dof))
    start_time = time.time()

    u = solve_linear_system(
        A_bc, f_bc, solver=solver, solver_parameters=solver_parameters
    )
    logging.debug("Solved system in %f sec" % (time.time() - start_time))

    function.set_vector(u)
    rhs_function = Function(matrix_assembler.mesh, matrix_assembler.function_size)
    rhs_function.set_vector(A.dot(u))

    return function, rhs_function


def nonlinear_solve(
    jacobian,
    residual,
    dirichlet_bcs,
    update_function=None,
    initial=None,
    tol=1e-10,
    solver="scipy_sparse",
    solver_parameters={},
):

    mesh = jacobian.mesh
    function_size = jacobian.function_size
    node_dofs = jacobian.node_dofs

    function = Function(mesh, function_size)
    if initial:
        function.vector = initial.vector.copy()

    rel_error = tol + 1

    u_dirichlet = get_dirichlet_vector(mesh, node_dofs, function_size, dirichlet_bcs)

    # phi0 = function.vector.copy()
    n_iter = 0

    while rel_error >= tol:
        old_vector = function.vector

        if update_function:
            update_function(mesh, function)

        start = time.time()
        logging.debug("Started assembling Jacobian matrix")
        A = jacobian.assemble()
        logging.debug(
            "Finished assembling Jacobian matrix in %fs" % (time.time() - start)
        )

        start = time.time()
        logging.debug("Started assembling residual vector")
        f = residual.assemble()
        logging.debug(
            "Finished assembling residual vector in %fs" % (time.time() - start)
        )

        A_bc = get_modified_matrix(A, mesh, node_dofs, function_size, dirichlet_bcs)
        constrained_dofs = get_constrained_dofs(
            mesh, node_dofs, function_size, dirichlet_bcs
        )

        update_dirichlet = np.zeros(u_dirichlet.shape)

        for n, constrained in enumerate(constrained_dofs):
            if constrained:
                update_dirichlet[n] = u_dirichlet[n] - old_vector[n]

        f_bc = f - A.dot(update_dirichlet)

        for n, constrained in enumerate(constrained_dofs):
            if constrained:
                f_bc[n] = update_dirichlet[n]

        update_vector = solve_linear_system(
            A_bc, f_bc, solver=solver, solver_parameters=solver_parameters
        )

        f_final = A.dot(update_vector)

        new_vector = old_vector + update_vector

        # u = Function(function.function_space)
        # u.vector = new_vector - phi0
        # ofile = VTKFile('out_%03d.vtk'%n_iter)
        # u.set_label('u')
        # ofile.write(function.function_space.mesh, [u])

        # rel_error = np.linalg.norm(old_vector-new_vector)
        # abs_error = np.linalg.norm(f_final)

        rel_error = np.max(np.abs(old_vector - new_vector))
        abs_error = np.max(np.abs(f_final))

        function.set_vector(new_vector)
        n_iter += 1

        # logging.info('#'+str(n_iter)+' rel_err: '+str(rel_error)+' abs_err: '+str(abs_error))
        logging.info("#%d rel_err: %.3e abs_err: %.3e" % (n_iter, rel_error, abs_error))

    residual_function = Function(mesh, function_size)
    residual_function.set_vector(f_final)

    return function, residual_function


def apply_bcs(matrix, rhs_vector, mesh, node_dofs, function_size, dirichlet_bcs):
    matrix = matrix.copy()
    rhs_vector = rhs_vector.copy()

    u_dirichlet = get_dirichlet_vector(mesh, node_dofs, function_size, dirichlet_bcs)
    rhs_vector = rhs_vector - matrix.dot(u_dirichlet)

    constrained_dofs = get_constrained_dofs(
        mesh, node_dofs, function_size, dirichlet_bcs
    )

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0):
                continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components:
                    continue
                for i in range(matrix.shape[0]):
                    matrix[i, I] = 0.0
                for i in range(matrix.shape[1]):
                    matrix[I, i] = 0.0

                matrix[I, I] = 1.0
                rhs_vector[I] = value[I_i]

    return matrix, rhs_vector


def get_modified_matrix(matrix, mesh, node_dofs, function_size, dirichlet_bcs):
    matrix = matrix.copy()

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0):
                continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components:
                    continue
                for i in range(matrix.shape[0]):
                    matrix[i, I] = 0.0
                for i in range(matrix.shape[1]):
                    matrix[I, i] = 0.0

                matrix[I, I] = 1.0

    return matrix


def get_dirichlet_vector(mesh, node_dofs, function_size, dirichlet_bcs):
    system_size = len(mesh.nodes) * function_size
    u_dirichlet = np.zeros((system_size, 1))

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0):
                continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components:
                    continue
                u_dirichlet[I] = value[I_i]

    return u_dirichlet


def get_constrained_dofs(mesh, node_dofs, function_size, dirichlet_bcs):
    system_size = len(mesh.nodes) * function_size
    result = [False for i in range(system_size)]

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0):
                continue
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components:
                    continue
                result[I] = True

    return result


# def apply_bcs(matrix, rhs_vector, function_space, dirichlet_bcs):
#     matrix = matrix.copy()
#     rhs_vector = rhs_vector.copy()

#     for bc in dirichlet_bcs:
#         for n in function_space.mesh.nodes:
#             if not bc.position_bool(n.coor): continue
#             # print(n.idx)
#             value = bc.value(n.coor)
#             for I_i, I in enumerate(function_space.node_dofs[n.idx]):
#                 # TODO: nonhomogeneous bcs: asymmetric matrix
#                 # for i in range(matrix.shape[0]):
#                 #     matrix[i,I] = 0.
#                 for i in range(matrix.shape[1]):
#                     matrix[I,i] = 0.

#                 matrix[I,I] = 1.
#                 rhs_vector[I] = value[I_i]

#     return matrix, rhs_vector


def solve_linear_system(A, b, solver="scipy_sparse", solver_parameters={}):
    if solver == "scipy_sparse":
        u = solve_scipy_sparse(A, b)
    elif solver == "petsc":
        from lyza.petsc import solve_petsc

        u = solve_petsc(A, b)
    else:
        raise Exception("Unknown solver: %s" % solver)

    return u


def solve_scipy_sparse(A, b):
    return spsolve(A, b).reshape(b.shape)
