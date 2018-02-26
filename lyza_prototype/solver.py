import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from lyza_prototype.function import Function

from lyza_prototype.form import AggregateBilinearForm, AggregateLinearForm


def solve(bilinear_form, linear_form, function, dirichlet_bcs):

    # A = csr_matrix(self.assemble_stiffness_matrix())
    V = function.function_space
    if not isinstance(bilinear_form, AggregateBilinearForm):
        bilinear_form = AggregateBilinearForm([bilinear_form])

    if not isinstance(linear_form, AggregateLinearForm):
        linear_form = AggregateLinearForm([linear_form])

    A = bilinear_form.assemble()
    A_bc = A.copy()

    f_bc = linear_form.assemble()

    apply_bcs(A_bc, f_bc, V, dirichlet_bcs)

    # import matplotlib
    # matplotlib.use('Qt4Agg')
    # import pylab as pl
    # pl.spy(A_bc)
    # pl.show()

    n_dof = A.shape[0]
    logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))
    u = spsolve(A_bc, f_bc).reshape(f_bc.shape)
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


    return function, rhs_function


def nonlinear_solve(lhs_derivative, lhs_eval, rhs, function, dirichlet_bcs, tol=1e-12):

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
        A_bc = A.copy()
        # n_dof = A.shape[0]

        f_bc = (lhs_eval+rhs).assemble()

        apply_bcs(A_bc, f_bc, V, dirichlet_bcs)

        logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))
        update_vector = spsolve(A_bc, f_bc).reshape(f_bc.shape)
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

    for bc in dirichlet_bcs:
        for n in function_space.mesh.nodes:
            if not bc.position_bool(n.coor): continue
            # print(n.idx)
            value = bc.value(n.coor)
            for I_i, I in enumerate(function_space.node_dofs[n.idx]):
                # TODO: nonhomogeneous bcs: asymmetric matrix
                # for i in range(matrix.shape[0]):
                #     matrix[i,I] = 0.
                for i in range(matrix.shape[1]):
                    matrix[I,i] = 0.

                matrix[I,I] = 1.
                rhs_vector[I] = value[I_i]

