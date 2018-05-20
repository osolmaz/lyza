import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import time

from lyza_prototype.function import Function
from lyza_prototype.form import AggregateBilinearForm, AggregateLinearForm
from lyza_prototype.vtk import VTKFile

def solve(matrix_assembler, vector_assembler, dirichlet_bcs, solver='scipy_sparse', solver_parameters={}):

    function = Function(matrix_assembler.mesh, matrix_assembler.function_size)
    # V = function.function_space

    A = matrix_assembler.assemble()
    f_bc = vector_assembler.assemble()

    A_bc, f_bc = apply_bcs(A, f_bc, matrix_assembler.mesh, matrix_assembler.node_dofs, matrix_assembler.function_size, dirichlet_bcs)

    n_dof = A.shape[0]
    logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))

    u = solve_linear_system(A_bc, f_bc, solver=solver, solver_parameters=solver_parameters)

    logging.debug('Solved')

    function.set_vector(u)
    rhs_function = Function(matrix_assembler.mesh, matrix_assembler.function_size)
    rhs_function.set_vector(A.dot(u))

    # import ipdb; ipdb.set_trace()

    # force_resultant = [0.,0.]
    # for bc in neumann_bcs:
    #     for n in self.nodes:
    #         if not bc.position_bool(n.coor): continue

    #         value = bc.value(n.coor)
    #         for n,I in enumerate(n.dofmap):
    #             force_resultant[n] += self.rhs_vector[I,0]

    # print(force_resultant)

    # import matplotlib
    # matplotlib.use('Qt4Agg')
    # import pylab as pl
    # pl.spy(A_bc)
    # pl.show()


    return function, rhs_function


def nonlinear_solve(
        lhs_derivative,
        lhs_eval,
        rhs,
        function,
        dirichlet_bcs,
        prev_sol_quantity_map,
        prev_sol_grad_quantity_map,
        tol=1e-10,
        solver='scipy_sparse',
        solver_parameters={}):

    function = function.copy()

    V = function.function_space
    n_dof = V.get_system_size()

    rel_error = tol + 1
    # function.set_vector(np.zeros((n_dof, 1)))

    old_vector = function.vector
    u_dirichlet = get_dirichlet_vector(V, dirichlet_bcs)

    phi0 = function.vector.copy()
    n_iter = 0

    while rel_error >= tol:
        old_vector = function.vector

        lhs_derivative.project_to_quadrature_points(function, prev_sol_quantity_map)
        lhs_derivative.project_gradient_to_quadrature_points(function, prev_sol_grad_quantity_map)

        lhs_eval.project_to_quadrature_points(function, prev_sol_quantity_map)
        lhs_eval.project_gradient_to_quadrature_points(function, prev_sol_grad_quantity_map)

        A = lhs_derivative.assemble()
        f = (lhs_eval+rhs).assemble()

        A_bc = get_modified_matrix(A, V, dirichlet_bcs)
        constrained_dofs = get_constrained_dofs(V, dirichlet_bcs)

        update_dirichlet = np.zeros((V.get_system_size(), 1))

        for n, constrained in enumerate(constrained_dofs):
            if constrained:
                update_dirichlet[n] = u_dirichlet[n] - old_vector[n]

        f_bc = f - A.dot(update_dirichlet)

        for n, constrained in enumerate(constrained_dofs):
            if constrained:
                f_bc[n] = update_dirichlet[n]

        update_vector = solve_linear_system(A_bc, f_bc, solver=solver, solver_parameters=solver_parameters)

        f_final = A.dot(update_vector)

        new_vector = old_vector + update_vector

        # u = Function(function.function_space)
        # u.vector = new_vector - phi0
        # ofile = VTKFile('out_%03d.vtk'%n_iter)
        # u.set_label('u')
        # ofile.write(function.function_space.mesh, [u])

        # rel_error = np.linalg.norm(old_vector-new_vector)
        # abs_error = np.linalg.norm(f_final)

        rel_error = np.max(np.abs(old_vector-new_vector))
        abs_error = np.max(np.abs(f_final))

        function.set_vector(new_vector)
        n_iter += 1

        logging.info('#'+str(n_iter)+' rel_err: '+str(rel_error)+' abs_err: '+str(abs_error))

    residual_function = Function(V)
    residual_function.set_vector(f_final)

    return function, residual_function

def apply_bcs(matrix, rhs_vector, mesh, node_dofs, function_size, dirichlet_bcs):
    matrix = matrix.copy()
    rhs_vector = rhs_vector.copy()

    u_dirichlet = get_dirichlet_vector(mesh, node_dofs, function_size, dirichlet_bcs)
    rhs_vector = rhs_vector - matrix.dot(u_dirichlet)

    constrained_dofs = get_constrained_dofs(mesh, node_dofs, function_size, dirichlet_bcs)

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0): continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components: continue
                for i in range(matrix.shape[0]):
                    matrix[i,I] = 0.
                for i in range(matrix.shape[1]):
                    matrix[I,i] = 0.

                matrix[I,I] = 1.
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
            if not bc.position_bool(n.coor, 0): continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components: continue
                for i in range(matrix.shape[0]):
                    matrix[i,I] = 0.
                for i in range(matrix.shape[1]):
                    matrix[I,i] = 0.

                matrix[I,I] = 1.

    return matrix


def get_dirichlet_vector(mesh, node_dofs, function_size, dirichlet_bcs):
    system_size = len(mesh.nodes)*function_size
    u_dirichlet = np.zeros((system_size, 1))

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0): continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components: continue
                u_dirichlet[I] = value[I_i]

    return u_dirichlet

def get_constrained_dofs(mesh, node_dofs, function_size, dirichlet_bcs):
    system_size = len(mesh.nodes)*function_size
    result = [False for i in range(system_size)]

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_size)

        for n in mesh.nodes:
            if not bc.position_bool(n.coor, 0): continue
            for I_i, I in enumerate(node_dofs[n.idx]):
                if not I_i in components: continue
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

def solve_linear_system(A, b, solver='scipy_sparse', solver_parameters={}):
    if solver == 'scipy_sparse':
        u = solve_scipy_sparse(A, b)
    elif solver == 'petsc':
        u = solve_petsc(A, b)
    else:
        raise Exception('Unknown solver: %s'%solver)

    return u


def solve_petsc(mat, vec):

    # A = PETSc.Mat().create(PETSc.COMM_SELF)
    csr_mat = csr_matrix(mat)
    A = PETSc.Mat().createAIJ(
        size=csr_mat.shape,
        csr=(csr_mat.indptr, csr_mat.indices,csr_mat.data))

    b = PETSc.Vec().create(PETSc.COMM_SELF)
    u = PETSc.Vec().create(PETSc.COMM_SELF)

    # A.setSizes(mat.shape)
    # A.setPreallocationNNZ(5)
    # A.setType("aij")

    b.setSizes(vec.shape[0])
    u.setSizes(vec.shape[0])

    A.setUp()
    b.setUp()
    u.setUp()

    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         comp = mat[i,j]

    #         if abs(comp) > 1e-12:
    #             A[i,j] = comp

    # start = time.time()
    # end = time.time()
    # logging.info("Solved in %f seconds"%(end - start))

    for i in range(vec.shape[0]):
        b[i] = vec[i]

    A.assemble()
    b.assemble()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('cg')
    # pc = ksp.getPC()
    # pc.setType('none')
    ksp.setFromOptions()

    ksp.solve(b, u)

    # import ipdb; ipdb.set_trace()
    return u.getArray().reshape(vec.shape)

def solve_scipy_sparse(A, b):
    return spsolve(A, b).reshape(b.shape)

