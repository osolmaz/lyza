from sympy import *
from lyza_prototype.partition_system import *
import itertools

lambda_ = Symbol('lambda')
mu = Symbol('mu')
E = Symbol('E')
nu = Symbol('nu')


def delta(i,j):
    if i == j:
        return 1
    else:
        return 0

def isotropic_elasticity_tensor(i,j,k,l):
    # index_map = [[1,4,6],[4,2,5],[6,5,3]]
    # return Symbol('C_%d%d'%(index_map[i][j]-1, index_map[k][l]-1))
    # return Symbol('C_%d%d%d%d'%(i+1,j+1,k+1,l+1))
    return lambda_*delta(i,j)*delta(k,l) + mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))

index_map = lambda i,j: 3*i + j

reshaped_tensor = zeros(9,9)

for i,j,k,l in itertools.product(range(3), range(3), range(3), range(3)):
    alpha = index_map(i,j)
    beta = index_map(k,l)

    reshaped_tensor[alpha,beta] += isotropic_elasticity_tensor(i,j,k,l)

# pprint(reshaped_tensor)


lambda_val = E*nu / (1+nu) / (1-2*nu)
mu_val = E/2/(1+nu)

reshaped_tensor = reshaped_tensor.subs([
    (lambda_, lambda_val),
    (mu, mu_val),
])

A_11, A_12, A_21, reshaped_tensor_symmetric = partition_matrix_sympy(reshaped_tensor, [], res_dofmap=[0,4,8,1,5,2])

# pprint(reshaped_tensor_symmetric)

A_11, A_12, A_21, A_22 = partition_matrix_sympy(
    reshaped_tensor_symmetric,
    [0,1,3],
    res_dofmap=[2,4,5])

plane_strain_tensor = simplify(A_11)
plane_stress_tensor = simplify(A_11 - A_12*A_22.inv()*A_21)
# pprint(simplify(plane_stress_tensor*(1-nu*nu)/E))


if __name__ == '__main__':

    print('3D tensor:')
    pprint(reshaped_tensor_symmetric)
    print(reshaped_tensor_symmetric)

    print('Plane stress tensor:')
    pprint(plane_stress_tensor)
    print(plane_stress_tensor)

    print('Plane strain tensor:')
    pprint(plane_strain_tensor)
    print(plane_strain_tensor)
