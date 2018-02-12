
class Node():
    def __init__(self, coor, idx, label=None):
        self.coor = coor
        self.label = label
        self.idx = idx

        # self.physical_dim = physical_dim
        # self.dofmap = []
        # for i in range(self.physical_dim):
        #     self.dofmap.append(idx*self.physical_dim+i)

