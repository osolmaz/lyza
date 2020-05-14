from sympy import *
import itertools


def permutation(lst):
    """\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    """
    if len(lst) > len(set(lst)):
        return 0
    parity = 1
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity


x, y = symbols("x y")
position = [x, y]

u = Matrix([sin(2 * pi * x) * cos(2 * pi * y), cos(2 * pi * x) * sin(2 * pi * y)])
# u = Matrix([sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*sin(2*pi*y)])
# u = Matrix([0,-x*(1-x)*y*(1-y)])


gradient = zeros(len(u), len(position))

for i, j in itertools.product(range(len(u)), range(len(position))):
    gradient[i, j] += (diff(u[i], position[j]) + diff(u[j], position[i])) / 2

gradient = simplify(gradient)
# pprint(gradient)

compatibility_test = (
    diff(diff(gradient[0, 0], y), y)
    + diff(diff(gradient[1, 1], x), x)
    - 2 * diff(diff(gradient[0, 1], x), y)
)


compatibility_test = simplify(compatibility_test)

print("%s ?= 0" % compatibility_test)
