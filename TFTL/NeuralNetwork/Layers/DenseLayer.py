import numpy as np

from TFTL.NeuralNetwork.Activations.LinearActivation import LinearActivation


class DenseLayer(object):
    def __init__(self, units=None, prev_layer=None, next_layer=None, input_shape=None, activation=LinearActivation()):
        # If the layer is the first layer, it must receive the input dimension of X. Otherwise it is of the previous layer.
        if prev_layer is not None:
            self.input_shape = prev_layer.M
        else:
            assert input_shape is not None
            self.input_shape = input_shape

        # Number of nodes in layer.
        self.M = units

        # Activation function of the layer.
        self.activation = activation

        # Layer closer to the input.
        self.__prev_layer = prev_layer
        # Layer closer to the output.
        self.__next_layer = next_layer

        # Regularization parameters.
        self.lambda1 = 0
        self.lambda2 = 0

    @property
    def prev_layer(self):
        return self.__prev_layer

    @prev_layer.getter
    def prev_layer(self):
        return self.__prev_layer

    @prev_layer.setter
    def prev_layer(self, val):
        self.__prev_layer = val
        self._set_weights()

    @property
    def next_layer(self):
        return self.__next_layer

    @next_layer.getter
    def next_layer(self):
        return self.__next_layer

    @next_layer.setter
    def next_layer(self, val):
        self.__next_layer = val
        self._set_weights()

    def feed_forward(self, Z_prev):
        self.Z = self.activation.function(np.matmul(self.W, Z_prev) + self.b)
        return self.Z

    def back_propagate(self, eta, **kwargs):
        # Output layer gradient.
        if self.next_layer is None:
            Z_hat = self.Z
            self.dH = Z_hat

        # Other layer gradients.
        else:
            self.dZ = np.matmul(self.next_layer.dH, self.next_layer.W.T)
            self.dH = self.dZ * self.activation.derivative(self.Z)

        self.dW = np.matmul(self.prev_layer.Z.T, self.dH) + self.lambda1 * np.sign(self.W) + self.lambda2 * self.W
        self.db = self.dH.sum(axis=0)
        self.W -= eta * self.dW
        self.b -= eta * self.db

    def _set_weights(self):
        self.W = np.random.randn(self.input_shape, self.M)
        self.b = np.random.randn(self.M)

    def __str__(self):
        return f"DenseLayer(units={self.M}, activation={self.activation})"
