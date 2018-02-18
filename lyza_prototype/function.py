import numpy as np

class Function:
    def __init__(self, function_space):
        self.function_space = function_space
        self.function_dimension = function_space.function_dimension

        n_dof = function_space.get_system_size()
        self.vector = np.zeros((n_dof,1))

    def set_vector(self, vector):
        self.vector = vector

    def set_label(self, label):
        self.label = label

    def get_node_val(self, idx):
        start_idx = idx*self.function_dimension
        end_idx = idx*self.function_dimension + self.function_dimension
        return self.vector[start_idx:end_idx]
