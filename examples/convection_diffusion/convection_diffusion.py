from lyza_prototype import *
from math import *

import itertools
import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 20

exact_solution = lambda x: [(2./pi)*atan(-0.5*x[0] + x[1]-0.25)]

exact_solution_deriv = lambda x: [[
    -1./(pi*((.5*x[0] - x[1] + .25)**2 + 1.)),
    2./(pi*((.5*x[0] - x[1] + .25)**2 + 1.))
]]

def force_function(x_coor):
    x = x_coor[0]
    y = x_coor[1]
    eps = 1.
    yex    = (2./pi)*atan(1./sqrt(eps)*(-0.5*x + y - 0.25))
    yex_x  = -1./(eps**0.5*pi*((x/2. - y + 0.25)**2./eps + 1.))
    yex_y  = 2./(eps**0.5*pi*((x/2. - y + 0.25)**2./eps + 1.))
    yex_xx = (x/2. - y + 0.25)/(eps**1.5*pi*((x/2. - y + 0.25)**2./eps + 1.)**2)
    yex_yy = (2.*x - 4.*y + 1.)/(eps**1.5*pi*((x/2. - y + 0.25)**2./eps + 1.)**2)
    return [-eps*(yex_xx+yex_yy) + 2.*yex_x+ 3.*yex_y +1.*yex]



class PoissonMatrix(lyza_prototype.ElementMatrix):

    def eval(self, K, N_p, B_p, det_jac, quad_point, function_dim, physical_dim, elem_dim, n_dof, n_node):
        for I,J,i in itertools.product(
                range(n_node),
                range(n_node),
                range(physical_dim)):
            # import ipdb; ipdb.set_trace()

            K[I, J] += B_p[I][i]*B_p[J][i]*det_jac

        return K

mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)


V = FunctionSpace(mesh, 1, 2, 1, 1)
u = Function(V)
a = BilinearForm(PoissonMatrix())
b_body_force = LinearForm(element_vectors.FunctionVector(force_function))


bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12

perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])


dirichlet_bcs = [DirichletBC(exact_solution, perimeter)]
# dirichlet_bcs = [DirichletBC(exact_solution, lambda x: True)]

u, f = solve(a, b_body_force, u, dirichlet_bcs)

ofile = VTKFile('out_poisson.vtk')

u.set_label('u')
f.set_label('f')

ofile.write(mesh, [u, f])

h_max = 1./RESOLUTION
n_node = len(mesh.nodes)
l2 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='l2')
linf = error.absolute_error(u, exact_solution, exact_solution_deriv, error='linf')
h1 = error.absolute_error(u, exact_solution, exact_solution_deriv, error='h1')

print(l2, linf, h1)

