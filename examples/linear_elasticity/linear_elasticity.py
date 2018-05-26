from lyza_prototype import *
import sympy as sp
import numpy as np
from plane_stress_strain import plane_stress_tensor, plane_strain_tensor
import itertools

import logging
logging.basicConfig(level=logging.INFO)

RESOLUTION = 20

E = 1000.
NU = 0.3

MU = mechanics.mu_from_E_nu(E, NU)
LAMBDA = mechanics.lambda_from_E_nu(E, NU)


# ELASTICITY_TENSOR = plane_stress_tensor
ELASTICITY_TENSOR = plane_strain_tensor

ELASTICITY_TENSOR = ELASTICITY_TENSOR.subs([(sp.Symbol('E'), E), (sp.Symbol('nu'), NU)])


class LinearElasticityAnalyticSolution(AnalyticSolution):
    def get_force_expression(self):
        f = sp.Matrix([0, 0])

        index_map = [[0,2],[2,1]]

        for i,j,k,l in itertools.product(range(2), range(2), range(2), range(2)):
            alpha = index_map[i][j]
            beta = index_map[k][l]

            f[i] += -ELASTICITY_TENSOR[alpha, beta] * sp.diff(sp.diff(self.u[k], self.position[l]), self.position[j])

        return f


analytic_sol_expr = lambda x: [sp.sin(2*sp.pi*x[0])*sp.sin(2*sp.pi*x[1]),
                               sp.sin(2*sp.pi*x[0])*sp.sin(2*sp.pi*x[1])]
# analytic_sol_expr = lambda x: [sp.sin(2*sp.pi*x[0])*sp.cos(2*sp.pi*x[1]),
#                                sp.sin(2*sp.pi*x[1])*sp.cos(2*sp.pi*x[0])]
# analytic_sol_expr = lambda x: [0, -x[0]*x[1]*(x[0] - 1)*(x[1] - 1)]

analytic_solution_obj = LinearElasticityAnalyticSolution(analytic_sol_expr, 2, 2)

analytic_solution = analytic_solution_obj.get_analytic_solution_function()
analytic_solution_gradient = analytic_solution_obj.get_gradient_function()
force_function = analytic_solution_obj.get_rhs_function()

bottom_boundary = lambda x, t: x[1] <= 1e-12
top_boundary = lambda x, t: x[1] >= 1. -1e-12
left_boundary = lambda x, t: x[0] <= 1e-12
right_boundary = lambda x, t: x[0] >= 1.-1e-12

perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

spatial_dimension = 2
function_size = 2
element_degree = 1
quadrature_degree = 1

if __name__ == '__main__':

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    a = matrix_assemblers.LinearElasticityMatrix(mesh, function_size)
    a.set_param_isotropic(LAMBDA, MU, plane_strain=True)

    b = vector_assemblers.FunctionVector(mesh, function_size)
    b.set_param(force_function, 0)

    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]

    u, f = solve(a, b, dirichlet_bcs)

    projector = iterators.SymmetricGradientProjector(mesh, function_size)
    projector.set_param(u, 'EPS', spatial_dimension)
    projector.execute()

    stress_calc = iterators.LinearStressCalculator(mesh, function_size)
    stress_calc.set_param_isotropic(LAMBDA, MU, plane_strain=True)
    stress_calc.init_stress_quantity(spatial_dimension)
    stress_calc.execute()

    stress = mesh.quantities['SIGV'].get_function()

    ofile = VTKFile('out_linear_elasticity.vtk')

    u.set_label('u')
    f.set_label('f')
    stress.set_label('stress')

    ofile.write(mesh, [u, f, stress])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, error='l2'))


