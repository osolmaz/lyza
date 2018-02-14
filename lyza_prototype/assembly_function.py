import numpy as np

class AssemblyFunction:
    def __init__(self, assembly, function_dimension):
        # self.function_space = function_space
        # self.function_dimension = self.function_space.get_dimension()
        self.function_dimension = function_dimension
        # self.degree = degree

        # self.quad_points = []
        # self.cell_quad_point_indices = []
        self.quad_point_dofmap = []
        self.assembly = assembly

        # current_idx = 0
        # for c in self.mesh.cells:
        #     points = c.get_quad_points(degree)
        #     # self.quad_points += points

        #     indices = []
        #     for p in points:
        #         indices.append(current_idx)
        #         current_idx += 1

        #     self.cell_quad_point_indices.append(indices)

        current_idx = 0
        for p in self.assembly.quad_points:
            dofmap = []
            for i in range(self.function_dimension):
                dofmap.append(current_idx)
                current_idx += 1
            self.quad_point_dofmap.append(dofmap)

        # self.vector = np.zeros(())
    # def set_val(self, point_idx, val):
