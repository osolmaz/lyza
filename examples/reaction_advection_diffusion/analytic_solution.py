from sympy import *
import itertools

t, x, y = symbols('t x y')
r = Symbol('r')
D_11, D_12, D_21, D_22 = symbols('D_00 D_01 D_10 D_11')
c_1, c_2 = symbols('c_0 c_1')
position = [x, y]

D = Matrix([[D_11, D_12],[D_21, D_22]])
c = Matrix([c_1, c_2])

u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)


f = 0

f += diff(u, t)

for i, j in itertools.product(range(2), range(2)):
    f += -D[i,j]*diff(diff(u, position[i]), position[j])

for i in range(2):
    f += c[i]*diff(u, position[i])

f += -r*u

f = factor(simplify(f))

pprint(f)
print(f)
