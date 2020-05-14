import sympy as sp
import numpy as np
import itertools


class AnalyticSolution:
    def __init__(self, u, n_eqn, n_dim, simplify=False):
        self.simplify = simplify
        self.n_eqn = n_eqn
        self.n_dim = n_dim

        x, y, z = sp.symbols("x y z")
        if n_dim == 1:
            self.position = [x]
        elif n_dim == 2:
            self.position = [x, y]
        elif n_dim == 3:
            self.position = [x, y, z]
        else:
            raise Exception("Invalid # of dimensions")

        self.u = u(self.position)
        if self.simplify:
            self.u = simplify(self.u)

    def get_rhs_expression(self):
        pass

    def get_rhs_function(self):
        f_expr = self.get_force_expression()
        if self.simplify:
            f_expr = simplify(f_expr)

        lambdas = [sp.lambdify(self.position, i) for i in f_expr]
        n_dim = self.n_dim

        def result(pos, t):

            # import ipdb; ipdb.set_trace()
            return [i(*pos[: self.n_dim]) for i in lambdas]

        return result

    def get_analytic_solution_function(self):
        # u_expr = self.get_ana_expression()

        lambdas = [sp.lambdify(self.position, i) for i in self.u]

        def result(pos, t):
            return [i(*pos[: self.n_dim]) for i in lambdas]

        return result

    def get_gradient_expression(self):
        gradient = sp.zeros(self.n_eqn, self.n_dim)

        for i, j in itertools.product(range(self.n_eqn), range(self.n_dim)):
            gradient[i, j] += sp.diff(self.u[i], self.position[j])

        return gradient

    def get_gradient_function(self):
        gradient_expr = self.get_gradient_expression()

        lambdas = []

        for i in range(self.n_eqn):
            row = []
            for j in range(self.n_dim):
                row.append(sp.lambdify(self.position, gradient_expr[i, j]))
            lambdas.append(row)

        def result(pos, t):
            return [[j(*pos[: self.n_dim]) for j in i] for i in lambdas]

        return result


def get_analytic_solution_vector(function_space, function, time=0):
    result = np.zeros((function_space.get_system_size(), 1))

    for n in function_space.mesh.nodes:
        analytic_val = function(n.coor, time)
        for n, dof in enumerate(function_space.node_dofs[n.idx]):
            result[dof] = analytic_val[n]

    return result
