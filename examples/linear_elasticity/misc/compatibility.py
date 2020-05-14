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


x, y, z = symbols("x y z")
position = [x, y, z]

u = Matrix([sin(2 * pi * x) * cos(2 * pi * y), cos(2 * pi * x) * sin(2 * pi * y), 0])
# u = Matrix([sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*sin(2*pi*y)])
# u = Matrix([0,-x*(1-x)*y*(1-y)])

gradient = zeros(3, 3)
for i, j in itertools.product(range(3), range(3)):
    gradient[i, j] += (diff(u[i], position[j]) + diff(u[j], position[i])) / 2

gradient = simplify(gradient)
pprint(gradient)

compatibility_test = 0

for i, j, k, l, m, n in itertools.product(
    range(3), range(3), range(3), range(3), range(3), range(3)
):
    compatibility_test += (
        permutation([i, j, k])
        * permutation([l, m, n])
        * diff(diff(gradient[j, m], position[n]), position[k])
    )
    # pprint(compatibility_test)

compatibility_test = simplify(compatibility_test)

print("%s ?= 0" % compatibility_test)
