

class Assembly:
    def __init__(self, elems):
        self.elems = elems

        self.cell_quad_point_indices = []
        self.quad_point_dofmap = []
        self.quad_points = []

        # current_idx = 0
        for e in self.elems:
            indices = []
            for p in e.quad_points:
                self.quad_points.append(p)
                indices.append(len(self.quad_points)-1)
                # current_idx += 1

            self.cell_quad_point_indices.append(indices)

        # current_idx = 0
        # for p in self.quad_points:
        #     dofmap = []
        #     for i in range(self.function_dimension):
        #         dofmap.append(current_idx)
        #         current_idx += 1
        #     self.quad_point_dofmap.append(dofmap)

        # self.
