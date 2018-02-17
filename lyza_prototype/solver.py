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

    n_dof = A.shape[0]

    for bc in dirichlet_bcs:
        for n in V.mesh.nodes:
            if not bc.position_bool(n.coor): continue

            value = bc.value(n.coor)
            for n,I in enumerate(V.node_dofs[n.idx]):
                for i in range(n_dof):
                    A_bc[I,i] = 0.
                    A_bc[i,I] = 0.

                A_bc[I,I] = 1.
                f_bc[I] = value[n]

    # import matplotlib
    # matplotlib.use('Qt4Agg')
    # import pylab as pl
    # pl.spy(A_bc)
    # pl.show()

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


def nonlinear_solve(lhs_derivative, lhs_eval, rhs, function, dirichlet_bcs):

    V = function.function_space

    if not isinstance(lhs, AggregateBilinearForm):
        lhs_bilinear = AggregateBilinearForm([lhs])

    # if not isinstance(lhs_eval, AggregateLinearForm):
    #     lhs_eval = AggregateLinearForm([lhs_eval])

    # if not isinstance(rhs, AggregateLinearForm):
    #     rhs = AggregateLinearForm([rhs])


    n_dof = A.shape[0]

    for bc in dirichlet_bcs:
        for n in V.mesh.nodes:
            if not bc.position_bool(n.coor): continue

            value = bc.value(n.coor)
            for n,I in enumerate(V.node_dofs[n.idx]):
                for i in range(n_dof):
                    A_bc[I,i] = 0.
                    A_bc[i,I] = 0.

                A_bc[I,I] = 1.
                f_bc[I] = value[n]

    # import matplotlib
    # matplotlib.use('Qt4Agg')
    # import pylab as pl
    # pl.spy(A_bc)
    # pl.show()

    logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))
    u = spsolve(A_bc, f_bc).reshape(f_bc.shape)
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


