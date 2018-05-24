import numpy as np

class Node():
    def __init__(self, coor, idx, label=None):
        if isinstance(coor, list):
            coor = np.array(coor).reshape(3,1)

        if coor.shape != (3,1):
            raise Exception('Invalid shape')

        self.coor = coor
        self.label = label
        self.idx = idx

        # self.spatial_dim = spatial_dim
        # self.dofmap = []
        # for i in range(self.spatial_dim):
        #     self.dofmap.append(idx*self.spatial_dim+i)

