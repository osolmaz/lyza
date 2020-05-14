from math import *
from lyza import *

import logging

logging.basicConfig(level=logging.INFO)

SPATIAL_DIMENSION = 2
FUNCTION_SIZE = 2
QUADRATURE_DEGREE = 1

L = 4.0
C = 1.0
P = 1.0

LAMBDA = 10000.0
MU = 1000.0

right_boundary = lambda x, t: x[0] >= L - 1e-12
left_boundary = lambda x, t: x[0] <= 1e-12

left_part = lambda x, t: x[0] <= L / 2.0
right_part = lambda x, t: x[0] >= L / 2.0

FUNCTION_VECTOR = lambda x, t: [0.0, -P / C]


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
    40, 10, [0.0, -C / 2.0], [L, -C / 2.0], [L, C / 2.0], [0.0, C / 2.0],
)
mesh.set_quadrature_degree(
    lambda c: QUADRATURE_DEGREE, SPATIAL_DIMENSION, domain=domain.AllDomain()
)

a1 = matrix_assemblers.LinearElasticityMatrix(mesh, FUNCTION_SIZE, domain=RightPart())
a2 = matrix_assemblers.LinearElasticityMatrix(mesh, FUNCTION_SIZE, domain=LeftPart())

a1.set_param_isotropic(10000.0, 1000.0, plane_stress=True)
a2.set_param_isotropic(1000.0, 100.0, plane_stress=True)

b = vector_assemblers.FunctionVector(mesh, FUNCTION_SIZE, domain=LeftEnd())
b.set_param(FUNCTION_VECTOR, 0)

dirichlet_bcs = [DirichletBC(lambda x, t: [0.0, 0.0], right_boundary)]

u, f = solve(a1 + a2, b, dirichlet_bcs)
ofile = VTKFile("out_different_materials.vtk")

u.set_label("u")
f.set_label("f")

ofile.write(mesh, [u, f])
