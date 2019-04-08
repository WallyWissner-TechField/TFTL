import numpy as np

from TFTL.NeuralNetwork.Activations.ActivationFunction import ActivationFunction


class ReLu(ActivationFunction):
    def __init__(self, alpha=0):
        super().__init__()

        self.alpha = alpha
        assert 0 <= self.alpha < 1

    def function(self, H):
        return H * (H > 0) + self.alpha * H * (H <= 0)

    def derivative(self, Z):
        return 1 * (Z > 0) + self.alpha * (Z <= 0)

    def __str__(self):
        return f"ReLu(alpha={self.alpha})"
