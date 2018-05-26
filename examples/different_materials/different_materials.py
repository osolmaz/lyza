from math import *
from lyza_prototype import *

import logging
logging.basicConfig(level=logging.INFO)

L = 4.
C = 1.
P = 1.

LAMBDA = 10000.
MU = 1000.

spatial_dimension = 2
function_size = 2
element_degree = 1
quadrature_degree = 1

right_boundary = lambda x, t: x[0] >= L-1e-12
left_boundary = lambda x, t: x[0] <= 1e-12

left_part = lambda x, t: x[0] <= L/2.
right_part = lambda x, t: x[0] >= L/2.

FUNCTION_VECTOR = lambda x, t: [0.,-P/C]

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
mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension, domain=domain.AllDomain())

a1 = matrix_assemblers.LinearElasticityMatrix(mesh, function_size, domain=RightPart())
a2 = matrix_assemblers.LinearElasticityMatrix(mesh, function_size, domain=LeftPart())

a1.set_param_isotropic(10000., 1000., plane_stress=True)
a2.set_param_isotropic(1000., 100., plane_stress=True)

b = vector_assemblers.FunctionVector(mesh, function_size, domain=LeftEnd())
b.set_param(FUNCTION_VECTOR, 0)

dirichlet_bcs = [DirichletBC(lambda x, t: [0.,0.], right_boundary)]

u, f = solve(a1+a2, b, dirichlet_bcs)
ofile = VTKFile('out_different_materials.vtk')

u.set_label('u')
f.set_label('f')

ofile.write(mesh, [u, f])
