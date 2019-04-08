import numpy as np


class Model(object):
    def __init__(self, phi=None, *args, **kwargs):
        self.X = None
        self.phi = phi if phi else lambda x: x
        self.PHI = None

    def _stack_ones(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.column_stack((ones, X))

    def D(self):
        print(f"PHI shape = {self.PHI.shape}")
        return self.PHI.shape[1]
