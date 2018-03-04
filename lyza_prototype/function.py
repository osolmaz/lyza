import numpy as np

class Function:
    def __init__(self, function_space):
        self.function_space = function_space
        self.function_size = function_space.function_size

        n_dof = function_space.get_system_size()
        self.vector = np.zeros((n_dof,1))
        self.label = None

    def set_vector(self, vector):
        self.vector = vector

    def set_label(self, label):
        self.label = label

    def get_node_val(self, idx):
        start_idx = idx*self.function_size
        end_idx = idx*self.function_size + self.function_size
        return self.vector[start_idx:end_idx]
