import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


def solve(bilinear_form, linear_form, function, dirichlet_bcs):

    # A = csr_matrix(self.assemble_stiffness_matrix())
    V = function.function_space
    A = bilinear_form.assemble(V)
    A_bc = np.array(A)

    f_bc = linear_form.assemble(V)

    n_dof = A.shape[0]

    for bc in dirichlet_bcs:
        for n in V.mesh.nodes:
            if not bc.position_bool(n.coor): continue

            value = bc.value(n.coor)
            for n,I in enumerate(n.dofmap):
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
    f_vector = A.dot(u)
    function.set_vector(u)

    # import ipdb; ipdb.set_trace()
