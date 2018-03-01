
class Node():
    def __init__(self, coor, idx, label=None):
        self.coor = coor
        self.label = label
        self.idx = idx

        # self.spatial_dim = spatial_dim
        # self.dofmap = []
        # for i in range(self.spatial_dim):
        #     self.dofmap.append(idx*self.spatial_dim+i)

