import numpy as np

from TFTL.NeuralNetwork.Activations.ActivationFunction import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()

    def function(self, H):
        return np.tanh(H)

    def derivative(self, Z):
        return 1 - Z**2

    def __str__(self):
        return "Tanh()"
