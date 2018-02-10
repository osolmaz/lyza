from math import *
import copy
import os

from quadmesh import QuadMesh
from mesh import DirichletBC, NeumannBC

import logging
logging.basicConfig(level=logging.INFO)


RESOLUTION = 20

PARAM = {
    'lambda': 10000.,
    'mu': 1000.,
    'resolution_x': RESOLUTION,
    'resolution_y': RESOLUTION,
    'p0': [0., 0.],
    'p1': [48., 44.],
    'p2': [48., 60.],
    'p3': [0., 44.],
    # 'p0': [0., 0.],
    # 'p1': [1., 0.],
    # 'p2': [1., 1.],
    # 'p3': [0., 1.],
}


mesh = QuadMesh(PARAM)

# print(mesh.elems[0].calc_stiffness_matrix())


# import ipdb; ipdb.set_trace()



left_boundary = lambda x: x[0] <= 1e-12
right_boundary = lambda x: x[0] >= 48. - 1e-12


dirichlet_bcs = [DirichletBC(lambda x: [0.,0.], left_boundary)]
neumann_bcs = [NeumannBC(lambda x: [0.,.5], right_boundary)]


mesh.solve(dirichlet_bcs, neumann_bcs=neumann_bcs)

mesh.write_vtk('out_quadmesh.vtk')



