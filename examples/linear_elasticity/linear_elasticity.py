from lyza_prototype import *
import sympy as sp
from plane_stress_strain import plane_stress_tensor, plane_strain_tensor
import itertools

import logging
logging.basicConfig(level=logging.INFO)

RESOLUTION = 20

E = 1000.
NU = 0.3

MU = E/(1.+NU)/2.
LAMBDA = E*NU/(1.+NU)/(1.-2.*NU)


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

bottom_boundary = lambda x: x[1] <= 1e-12
top_boundary = lambda x: x[1] >= 1. -1e-12
left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 1.-1e-12

perimeter = join_boundaries([bottom_boundary, top_boundary, left_boundary, right_boundary])

if __name__ == '__main__':

    mesh = meshes.UnitSquareMesh(RESOLUTION, RESOLUTION)

    spatial_dimension = 2
    function_size = 2
    element_degree = 1
    quadrature_degree = 1

    V = FunctionSpace(mesh, function_size, spatial_dimension, element_degree)
    u = Function(V)
    a = BilinearForm(V, V, bilinear_interfaces.IsotropicLinearElasticity(LAMBDA, MU, plane_strain=True), quadrature_degree)
    b_body_force = LinearForm(V, linear_interfaces.FunctionInterface(force_function), quadrature_degree)


    dirichlet_bcs = [DirichletBC(analytic_solution, perimeter)]
    # dirichlet_bcs = [DirichletBC(analytic_solution, lambda x: True)]

    u, f = solve(a, b_body_force, u, dirichlet_bcs)

    ofile = VTKFile('out_linear_elasticity.vtk')

    u.set_label('u')
    f.set_label('f')

    ofile.write(mesh, [u, f])

    print('L2 Error: %e'%error.absolute_error(u, analytic_solution, analytic_solution_gradient, quadrature_degree, error='l2'))


