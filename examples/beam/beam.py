from math import *
from pylyza import *

import logging
logging.basicConfig(level=logging.INFO)

L = 4.
C = 1.
P = 1.

LAMBDA = 10000.
MU = 1000.

E = MU*(3.*LAMBDA+2.*MU)/(LAMBDA+MU)
NU = LAMBDA/2./(LAMBDA+MU)
I = 1./12.*C*C*C

def exact_solution(coor):
    x = coor[0]
    y = coor[1]

    # q = P/C
    # delta = 5./24.*q*L*L*L*L/E/I*(1. + 12./5.*C*C/L/L*(4./5. + NU/2.))

    # u = q/2./E/I*((L*L*x - x*x*x/3.)*y + x*(2./3.*y*y*y - 2./5.*C*C*y) \
    #               + NU*x*(1./3.*y*y*y - C*C*y + 2./3.*C*C*C))

    # v = -q/2./E/I*(y*y*y*y/12. - C*C*y*y/2. + 2./3.*C*C*C*y \
    #                + NU*((L*L - x*x)*y*y/2. + y*y*y*y/6. - 1./5.*C*C*y*y)) \
    #                -1/2./E/I*(L*L*x*x/2. - x*x*x*x/12. - 1./5.*C*C*x*x\
    #                           +(1. + 1./2.*NU)*C*C*x*x)\
    #                + delta

    u = -P*x*x*y/2./E/I - NU*P*y*y*y/6./E/I + P*y*y*y/6./I/MU \
        + (P*L*L/2./E/I - P*C*C/2./I/MU)*y
    v = NU*P*x*y*y/2./E/I + P*x*x*x/6./E/I - P*L*L*x/2./E/I + P*L*L*L/3./E/I

    return [u, -v]

class LeftEnd(Domain):
    def is_subset(self, cell, is_boundary):
        is_in = not (False in [left_boundary(node.coor) for node in cell.nodes])

        # if is_in and is_boundary:
        #     print('ASDASD')

        return is_in and is_boundary

right_boundary = lambda x: x[0] >= L-1e-12
left_boundary = lambda x: x[0] <= 1e-12

mesh = meshes.QuadMesh(
    40,
    10,
    [0., -C/2.],
    [L, -C/2.],
    [L, C/2.],
    [0., C/2.],
)

V = FunctionSpace(mesh, 2, 2, 1, 1)
u = Function(V)
a = BilinearForm(element_matrices.LinearElasticityMatrix(LAMBDA, MU))
b_neumann = LinearForm(element_vectors.FunctionVector(lambda x: [0.,-P/C]), domain=LeftEnd())


dirichlet_bcs = [DirichletBC(lambda x: [0.,0.], right_boundary)]
# dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

u, f = solve(a, b_neumann, u, dirichlet_bcs)
ofile = VTKFile('out_beam.vtk')

u.set_label('displacement')
f.set_label('force')

ofile.write(mesh, [u, f])


# print(exact_solution([0.,-C/2.]))


