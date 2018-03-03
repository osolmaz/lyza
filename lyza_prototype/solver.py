import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import time

from lyza_prototype.function import Function
from lyza_prototype.form import AggregateBilinearForm, AggregateLinearForm


def solve(bilinear_form, linear_form, function, dirichlet_bcs, solver='scipy_sparse', solver_parameters={}):

    V = function.function_space
    # if not isinstance(bilinear_form, AggregateBilinearForm):
    #     bilinear_form = AggregateBilinearForm([bilinear_form])

    # if not isinstance(linear_form, AggregateLinearForm):
    #     linear_form = AggregateLinearForm([linear_form])

    A = bilinear_form.assemble()
    f_bc = linear_form.assemble()

    A_bc, f_bc = apply_bcs(A, f_bc, V, dirichlet_bcs)

    n_dof = A.shape[0]
    logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))

    u = solve_linear_system(A_bc, f_bc, solver=solver, solver_parameters=solver_parameters)

    logging.debug('Solved')

    function.set_vector(u)
    rhs_function = Function(V)
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


def nonlinear_solve(lhs_derivative, lhs_eval, rhs, function, dirichlet_bcs, tol=1e-12, solver='scipy_sparse', solver_parameters={}):

    V = function.function_space
    n_dof = V.get_system_size()

    # if not isinstance(lhs, AggregateBilinearForm):
    #     lhs_bilinear = AggregateBilinearForm([lhs])


    rel_error = 1.
    function.set_vector(np.zeros((n_dof, 1)))

    while rel_error >= tol:
        lhs_derivative.set_prev_sol(function)
        lhs_eval.set_prev_sol(function)
        A = lhs_derivative.assemble()
        # A_bc = A.copy()
        # n_dof = A.shape[0]

        f = (lhs_eval+rhs).assemble()

        A_bc, f_bc = apply_bcs(A, f, V, dirichlet_bcs)

        # logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))
        update_vector = solve_linear_system(A_bc, f_bc, solver=solver, solver_parameters=solver_parameters)

        old_vector = function.vector
        new_vector = old_vector + update_vector
        # import ipdb; ipdb.set_trace()
        rel_error = np.linalg.norm(old_vector-new_vector)
        function.set_vector(new_vector)
        logging.info(rel_error)

    rhs_function = Function(V)
    rhs_function.set_vector(A.dot(new_vector))

    return function, rhs_function

def apply_bcs(matrix, rhs_vector, function_space, dirichlet_bcs):
    matrix = matrix.copy()
    rhs_vector = rhs_vector.copy()

    u_dirichlet = get_dirichlet_vector(function_space, dirichlet_bcs)
    rhs_vector = rhs_vector - matrix.dot(u_dirichlet)

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_space.function_size)

        for n in function_space.mesh.nodes:
            if not bc.position_bool(n.coor): continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(function_space.node_dofs[n.idx]):
                if not I_i in components: continue
                for i in range(matrix.shape[0]):
                    matrix[i,I] = 0.
                for i in range(matrix.shape[1]):
                    matrix[I,i] = 0.

                matrix[I,I] = 1.
                rhs_vector[I] = value[I_i]

    return matrix, rhs_vector

def get_dirichlet_vector(function_space, dirichlet_bcs):
    u_dirichlet = np.zeros((function_space.get_system_size(), 1))

    for bc in dirichlet_bcs:
        if bc.components:
            components = bc.components
        else:
            components = range(function_space.function_size)

        for n in function_space.mesh.nodes:
            if not bc.position_bool(n.coor): continue
            value = bc.value(n.coor)
            for I_i, I in enumerate(function_space.node_dofs[n.idx]):
                if not I_i in components: continue
                u_dirichlet[I] = value[I_i]

    return u_dirichlet

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

