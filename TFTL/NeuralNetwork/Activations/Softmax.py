import numpy as np

from TFTL.NeuralNetwork.Activations.ActivationFunction import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def function(self, H):
        shiftH = H - np.max(H)
        exps = np.exp(shiftH)
        return exps / np.sum(exps)

    def derivative(self, Z):
        return Z * (1 - Z)

    def __str__(self):
        return "Softmax()"
