import numpy as np
from math import sqrt

def inverse(J):

    if J.shape[0] == J.shape[1]:
        return np.linalg.inv(J)
    else:
        return np.linalg.inv(J.transpose().dot(J)).dot(J.transpose())

def determinant(J):
    if J.shape[0] == J.shape[1]:
        return np.linalg.det(J)
    else:
        return sqrt(np.linalg.det(J.transpose().dot(J)))
