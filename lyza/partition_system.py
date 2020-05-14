import numpy as np
import sympy as sp


def partition_system(A, b, rem_dofmap, sort=True):
    A_uu, A_ku, A_uk, A_kk = partition_matrix(A, rem_dofmap, sort=sort)
    b_u, b_k = partition_vector(b, rem_dofmap, sort=sort)
    return A_uu, A_ku, A_uk, A_kk, b_u, b_k


def partition_vector(b, rem_dofmap, sort=True):
    if sort:
        rem_dofmap = sorted(rem_dofmap)
    res_dofmap = list(range(len(b)))
    for i in sorted(rem_dofmap, reverse=True):
        del res_dofmap[i]

    b_u = np.zeros((len(rem_dofmap), 1))
    b_k = np.zeros((len(res_dofmap), 1))

    for n_i, idx_i in enumerate(rem_dofmap):
        b_u[n_i] = b[idx_i]

    for n_i, idx_i in enumerate(res_dofmap):
        b_k[n_i] = b[idx_i]

    return b_u, b_k


def partition_matrix(A, rem_dofmap, sort=True):
    if sort:
        rem_dofmap = sorted(rem_dofmap)
    res_dofmap = list(range(A.shape[0]))

    for i in sorted(rem_dofmap, reverse=True):
        del res_dofmap[i]

    A_uu = np.zeros((len(rem_dofmap), len(rem_dofmap)))
    A_ku = np.zeros((len(res_dofmap), len(rem_dofmap)))
    A_uk = np.zeros((len(rem_dofmap), len(res_dofmap)))
    A_kk = np.zeros((len(res_dofmap), len(res_dofmap)))

    for n_i, idx_i in enumerate(rem_dofmap):
        for n_j, idx_j in enumerate(rem_dofmap):
            A_uu[n_i][n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(res_dofmap):
        for n_j, idx_j in enumerate(rem_dofmap):
            A_ku[n_i][n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(rem_dofmap):
        for n_j, idx_j in enumerate(res_dofmap):
            A_uk[n_i][n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(res_dofmap):
        for n_j, idx_j in enumerate(res_dofmap):
            A_kk[n_i][n_j] = A[idx_i, idx_j]

    return A_uu, A_ku, A_uk, A_kk


def partition_system_sympy(A, b, rem_dofmap, res_dofmap=None, sort=True):
    A_uu, A_uk, A_ku, A_kk = partition_matrix_sympy(
        A, rem_dofmap, res_dofmap=res_dofmap, sort=sort
    )
    b_u, b_k = partition_vector_sympy(b, rem_dofmap, res_dofmap=res_dofmap, sort=sort)
    return A_uu, A_uk, A_ku, A_kk, b_u, b_k


def partition_vector_sympy(b, rem_dofmap, res_dofmap=None, sort=True):
    if sort:
        rem_dofmap = sorted(rem_dofmap)

    if not res_dofmap:
        res_dofmap = list(range(len(b)))
        for i in sorted(rem_dofmap, reverse=True):
            del res_dofmap[i]

    b_u = sp.zeros(len(rem_dofmap), 1)
    b_k = sp.zeros(len(res_dofmap), 1)

    for n_i, idx_i in enumerate(rem_dofmap):
        b_u[n_i] = b[idx_i]

    for n_i, idx_i in enumerate(res_dofmap):
        b_k[n_i] = b[idx_i]

    return b_u, b_k


def partition_matrix_sympy(A, rem_dofmap, res_dofmap=None, sort=True):
    if sort:
        rem_dofmap = sorted(rem_dofmap)
    if not res_dofmap:
        res_dofmap = list(range(A.shape[0]))

        for i in sorted(rem_dofmap, reverse=True):
            del res_dofmap[i]

    A_uu = sp.zeros(len(rem_dofmap), len(rem_dofmap))
    A_ku = sp.zeros(len(res_dofmap), len(rem_dofmap))
    A_uk = sp.zeros(len(rem_dofmap), len(res_dofmap))
    A_kk = sp.zeros(len(res_dofmap), len(res_dofmap))

    for n_i, idx_i in enumerate(rem_dofmap):
        for n_j, idx_j in enumerate(rem_dofmap):
            A_uu[n_i, n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(res_dofmap):
        for n_j, idx_j in enumerate(rem_dofmap):
            A_ku[n_i, n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(rem_dofmap):
        for n_j, idx_j in enumerate(res_dofmap):
            A_uk[n_i, n_j] = A[idx_i, idx_j]

    for n_i, idx_i in enumerate(res_dofmap):
        for n_j, idx_j in enumerate(res_dofmap):
            A_kk[n_i, n_j] = A[idx_i, idx_j]

    return A_uu, A_uk, A_ku, A_kk


if __name__ == "__main__":
    size = 6

    A = np.zeros((size, size))
    b = np.zeros((size, 1))

    for i in range(size):
        b[i] = i + 1
        for j in range(size):
            A[i, j] = i * size + j + 1

    A_uu, A_ku, A_uk, A_kk, b_u, b_k = partition_system(A, b, [2, 3])

    print("A = ")
    print(A)
    print()

    print("b = ")
    print(b)
    print()

    print("A_uu = ")
    print(A_uu)
    print()

    print("A_ku = ")
    print(A_ku)
    print()

    print("A_uk = ")
    print(A_uk)
    print()

    print("A_kk = ")
    print(A_kk)
    print()

    print("b_u = ")
    print(b_u)
    print()

    print("b_k = ")
    print(b_k)
    print()
