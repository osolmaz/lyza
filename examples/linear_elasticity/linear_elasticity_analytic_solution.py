from plane_stress_strain import plane_stress_tensor, plane_strain_tensor
from sympy import *
import itertools

x, y = symbols('x y')

position = [x,y]

u = Matrix([sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y)])
# u = Matrix([0,-x*(1-x)*y*(1-y)])


f = Matrix([0, 0])
gradient = Matrix([[0, 0],[0,0]])

index_map = [[0,2],[2,1]]

# elasticity_tensor = plane_stress_tensor
elasticity_tensor = plane_strain_tensor

for i,j,k,l in itertools.product(range(2), range(2), range(2), range(2)):
    alpha = index_map[i][j]
    beta = index_map[k][l]

    f[i] += -elasticity_tensor[alpha, beta] * diff(diff(u[k], position[l]), position[j])

for i,j in itertools.product(range(2), range(2)):
    gradient[i,j] += diff(u[i], position[j])


print('C')
pprint(simplify(elasticity_tensor))
print(simplify(elasticity_tensor))

print('Analytic solution')
pprint(simplify(u))
print(simplify(u))

print('Gradient')
pprint(simplify(gradient))
print(simplify(gradient))

print('Force')
pprint(simplify(f))
print(simplify(f))


# import ipdb; ipdb.set_trace()
