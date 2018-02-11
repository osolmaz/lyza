import numpy as np

class Function:
    def __init__(self, function_space):
        self.function_space = function_space

        n_dof = function_space.get_system_size()
        self.vector = np.zeros((n_dof,1))
        # pass

    def set_vector(self, vector):
        self.vector = vector
