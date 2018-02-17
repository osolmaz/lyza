from math import *
from lyza_prototype import *

import logging
logging.basicConfig(level=logging.INFO)

L = 4.
C = 1.
P = 1.

LAMBDA = 10000.
MU = 1000.

right_boundary = lambda x: x[0] >= L-1e-12
left_boundary = lambda x: x[0] <= 1e-12

left_part = lambda x: x[0] <= L/2.
right_part = lambda x: x[0] >= L/2.

class LeftEnd(Domain):
    def is_subset(self, cell):
        return cell.all_nodes_in(left_boundary) and cell.is_boundary

class LeftPart(Domain):
    def is_subset(self, cell):
        return cell.all_nodes_in(left_part) and not cell.is_boundary

class RightPart(Domain):
    def is_subset(self, cell):
        return cell.all_nodes_in(right_part) and not cell.is_boundary

mesh = meshes.QuadMesh(
    40,
    10,
    [0., -C/2.],
    [L, -C/2.],
    [L, C/2.],
    [0., C/2.],
)

physical_dimension = 2
function_dimension = 2
element_degree = 1
quadrature_degree = 1

V = FunctionSpace(mesh, function_dimension, physical_dimension, element_degree)

u = Function(V)

matrix1 = element_matrices.LinearElasticityMatrix(10000., 1000.)
matrix2 = element_matrices.LinearElasticityMatrix(1000., 100.)

a1 = BilinearForm(V, V, matrix1, quadrature_degree, domain=RightPart())
a2 = BilinearForm(V, V, matrix2, quadrature_degree, domain=LeftPart())


b_neumann = LinearForm(
    V,
    element_vectors.FunctionElementVector(lambda x: [0.,-P/C]),
    quadrature_degree,
    domain=LeftEnd())

dirichlet_bcs = [DirichletBC(lambda x: [0.,0.], right_boundary)]

u, f = solve(a1+a2, b_neumann, u, dirichlet_bcs)
ofile = VTKFile('out_different_materials.vtk')

u.set_label('displacement')
f.set_label('force')

ofile.write(mesh, [u, f])

# print(exact_solution([0.,-C/2.]))


