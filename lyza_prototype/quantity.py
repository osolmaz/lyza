
import numpy as np

class Quantity:
    def __init__(self, shape, n_quantity):
        self.shape = shape
        self.vectors = [np.zeros(shape) for i in range(n_quantity)]


