import numpy as np

from TFTL.NeuralNetwork.Activations.ActivationFunction import ActivationFunction


class LinearActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

    def function(self, H):
        return H

    def derivative(self, Z):
        return np.ones((Z.shape[1], 1))

    def __str__(self):
        return "LinearActivation()"
