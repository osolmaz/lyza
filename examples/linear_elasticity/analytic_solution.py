from plane_stress_strain import plane_stress_tensor, plane_strain_tensor
from sympy import *
import itertools

x, y = symbols('x y')
position = [x,y]


def get_force_expression(u, plane_stress=True):
    f = Matrix([0, 0])
    gradient = Matrix([[0, 0],[0,0]])

    index_map = [[0,2],[2,1]]

    if plane_stress:
        elasticity_tensor = plane_stress_tensor
    else:
        elasticity_tensor = plane_strain_tensor

    for i,j,k,l in itertools.product(range(2), range(2), range(2), range(2)):
        alpha = index_map[i][j]
        beta = index_map[k][l]

        f[i] += -elasticity_tensor[alpha, beta] * diff(diff(u[k], position[l]), position[j])

    return f

def get_gradient_expression(u):
    gradient = Matrix([[0, 0],[0,0]])
    for i,j in itertools.product(range(2), range(2)):
        gradient[i,j] += diff(u[i], position[j])

    return gradient

def get_analytic_solution_function(u):
    u1 = lambdify((x,y), u[0])
    u2 = lambdify((x,y), u[1])
    def result(pos):
        return [
            u1(pos[0],pos[1]),
            u2(pos[0],pos[1]),
        ]

    return result

def get_force_function(u, E, nu, plane_stress=True):
    f_expr = get_force_expression(u, plane_stress=plane_stress)
    f_expr = simplify(f_expr)
    f_expr = f_expr.subs([(Symbol('E'), E), (Symbol('nu'), nu)])
    f1 = lambdify((x,y), f_expr[0])
    f2 = lambdify((x,y), f_expr[1])

    def result(pos):
        # import ipdb; ipdb.set_trace()
        return [
            f1(pos[0],pos[1]),
            f2(pos[0],pos[1]),
        ]

    return result

def get_gradient_function(u):
    gradient_expr = get_gradient_expression(u)

    gradient11 = lambdify((x,y), gradient_expr[0,0])
    gradient12 = lambdify((x,y), gradient_expr[0,1])
    gradient21 = lambdify((x,y), gradient_expr[1,0])
    gradient22 = lambdify((x,y), gradient_expr[1,1])
    def result(pos):
        return [
            [gradient11(pos[0],pos[1]), gradient12(pos[0],pos[1])],
            [gradient21(pos[0],pos[1]), gradient22(pos[0],pos[1])],
        ]

    return result


if __name__ == '__main__':

    PLANE_STRESS = True

    # u = Matrix([sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y)])
    u = Matrix([sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*sin(2*pi*y)])
    # u = Matrix([0,-x*(1-x)*y*(1-y)])


    # force = get_force_function(u, 1., 0.3, plane_stress=PLANE_STRESS)
    # analytic_solution = get_analytic_solution_function(u)
    # gradient = get_gradient_function(u)

    # test_point = [0.1,0.1]
    # print(force(test_point))
    # print(analytic_solution(test_point))
    # print(gradient(test_point))

    # print('C')
    # pprint(simplify(elasticity_tensor))
    # print(simplify(elasticity_tensor))

    u_expr = simplify(u)
    print('Analytic solution')
    pprint(u_expr)
    print(u_expr)

    gradient_expr = simplify(get_gradient_expression(u))
    print('Gradient')
    pprint(gradient_expr)
    print(gradient_expr)

    force_expr = simplify(get_force_expression(u, plane_stress=PLANE_STRESS))
    print('Force')
    pprint(force_expr)
    print(force_expr)


    # import ipdb; ipdb.set_trace()
