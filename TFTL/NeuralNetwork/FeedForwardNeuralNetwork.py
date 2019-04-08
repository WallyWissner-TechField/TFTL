import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FeedForwardNeuralNetwork(object):
    def __init__(self, loss, optimizer, metrics, lambda1, lambda2):
        # Objective function.
        self.losses = {
            'entropy': self._cross_entropy,
            'ols': self._OLS
        }
        self.loss = self.losses[loss]

        # Optimization method.
        self.optimizers = {
            'sgd': 0,  # Stochastic gradient descent.
        }
        self.optimizer = self.optimizers[optimizer]

        # Regularization parameters.
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Network layers.
        self.layers = []

    def fit(self, X, Y, epochs, eta, show=False, **kwargs):
        # UNNEEDED
        # Set the network input to X.
        self.layers[0].Z = X

        epochs = int(epochs)

        J_list = [0] * epochs

        for epoch in range(epochs):
            self._feed_forward(X)
            self._back_propagate(X, Y, eta, **kwargs)

            J_list[epoch] = self.loss() \
                            + self.lambda1 * np.sum(np.sum(abs(layer.W)) for layer in self.layers) \
                            + self.lambda2/2 * np.sum(np.sum(layer.W**2) for layer in self.layers)

        if show:
            plt.figure()
            plt.plot(J_list)
            plt.show()

    def predict(self, X):
        return self._feed_forward(X)

    def predict_category(self, X):
        return np.argmax(self.predict(X))

    @staticmethod
    def _cross_entropy(Y, P_hat):
        return -np.sum(Y * np.log(P_hat))

    @staticmethod
    def _OLS(Y, Y_hat):
        return np.sum(np.matmul((Y - Y_hat).T, (Y - Y_hat)))

    @staticmethod
    def _one_hot_encode(y, K):
        N = len(y)
        Y = np.zeros((N, K))
        for i in range(N):
            Y[i, y[i]] = 1
        return Y

    def _feed_forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.feed_forward(Z)
        return Z

    def _back_propagate(self, X, Y, eta, **kwargs):
        P_hat = self.predict(X)
        Z_ahead = P_hat - Y
        for layer in self.layers[::-1]:
            layer.back_propogate(eta, **kwargs)

    def add(self, layer):
        # Set regularization parameters.
        layer.lambda1 = self.lambda1
        layer.lambda2 = self.lambda2

        # The first added layer has to specify input dimension.
        if len(self.layers) == 0:
            assert layer.input_dimension is not None

        if len(self.layers) > 0:
            # Set the added layer's previous layer to the network's previous layer.
            layer.prev_layer = self.layers[-1]
            # Set the network's last layer's next layer to the added layer.
            self.layers[-1].next_layer = layer

        # Add the layer to the network.
        self.layers.append(layer)

    def _N(self):
        return self.layers[1].Z.shape[0]

    def _D(self):
        return self.layers[1].Z.shape[1]

    def _K(self):
        return self.layers[-1].Z.shape[1]

    def accuracy(self, X, y):
        y_hat = self.predict_category(X)
        return np.mean(y == y_hat)

    def confusion(self, X, y):
        K = self._K()
        y_hat = self.predict_category(X)
        N = y.shape[0]
        Y = self._one_hot_encode(y, K)
        Y_hat = self._one_hot_encode(y_hat, K)
        df = pd.DataFrame(Y.T @ Y_hat)
        df.columns = range(K)
        df.index = range(K)
        return df / N

    def __str__(self):
        return f"FeedForwardNeuralNetwork(layers={self.layers})"
