from petsc4py import PETSc

def solve_petsc(mat, vec):

    # A = PETSc.Mat().create(PETSc.COMM_SELF)
    csr_mat = csr_matrix(mat)
    A = PETSc.Mat().createAIJ(
        size=csr_mat.shape,
        csr=(csr_mat.indptr, csr_mat.indices,csr_mat.data))

    b = PETSc.Vec().create(PETSc.COMM_SELF)
    u = PETSc.Vec().create(PETSc.COMM_SELF)

    # A.setSizes(mat.shape)
    # A.setPreallocationNNZ(5)
    # A.setType("aij")

    b.setSizes(vec.shape[0])
    u.setSizes(vec.shape[0])

    A.setUp()
    b.setUp()
    u.setUp()

    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         comp = mat[i,j]

    #         if abs(comp) > 1e-12:
    #             A[i,j] = comp

    # start = time.time()
    # end = time.time()
    # logging.info("Solved in %f seconds"%(end - start))

    for i in range(vec.shape[0]):
        b[i] = vec[i]

    A.assemble()
    b.assemble()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('cg')
    # pc = ksp.getPC()
    # pc.setType('none')
    ksp.setFromOptions()

    ksp.solve(b, u)

    # import ipdb; ipdb.set_trace()
    return u.getArray().reshape(vec.shape)
