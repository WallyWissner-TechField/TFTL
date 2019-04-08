import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import TFTL.NeuralNetwork.Optimizers as op


class SimpleFFNN(object):
    def __init__(self, loss, optimizer, metrics, lambda1, lambda2):
        # Objective function.
        losses = {
            'entropy': self._cross_entropy,
            'ols': self._OLS
        }
        self.loss = loss
        self.LOSS = losses[loss]

        # Optimization method.
        eta = 1e-3
        optimizers = {
            'sgd': op.StochasticGradientDescent,  # Stochastic gradient descent.
            'adagrad': 0,
            'rmsprop': 0,
            'momentum': 0,
            'nesterov': 0,
            'rmsprop_m': op.RMSPropMomentum,
            'adam': op.Adam,
        }
        self.optimizer = optimizer
        self.OPTIMIZER_W = optimizers[optimizer](eta)
        self.OPTIMIZER_b = optimizers[optimizer](eta)

        # Regularization parameters.
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Network layers.
        self.A = {}   # Activation functions.
        self.W = {}   # Weight matrices.
        self.dW = {}
        self.b = {}   # Bias weight vectors.
        self.db = {}
        self.H = {}   # Logits.
        self.dH = {}
        self.Z = {}   # Inputs.
        self.dZ = {}

    def fit(self, X, Y, epochs, eta, batch_size=None, show=False, show_frequency=None, **kwargs):
        self.OPTIMIZER_W.eta = eta
        self.OPTIMIZER_b.eta = eta

        # If batch size is not given use all of the data.
        N = X.shape[0]
        batch_size = batch_size if batch_size is not None else N
        try:
            assert batch_size <= N
        except AssertionError:
            raise ValueError("Batch size cannot be larger than data to be fit.")

        epochs = int(epochs)
        show_frequency = show_frequency if show_frequency is not None else max(1, epochs // 20)

        J_list = [0] * epochs

        for epoch in tqdm(range(epochs)):

            # Shuffle data.
            shuffle = np.random.permutation(range(N))
            X_shuffled = X[shuffle, :]
            Y_shuffled = Y[shuffle, :]

            # Separate the shuffled data into batches of size batch_size.
            batches = [range(i * batch_size, (i + 1) * batch_size) for i in range(N // batch_size)]

            # Perform gradient descent on each batch.
            for i, batch in enumerate(batches):
                X_batch = X_shuffled[batch, :]
                Y_batch = Y_shuffled[batch, :]

                self._feed_forward(X_batch)
                self._back_propagate(Y=Y_batch, eta=eta, epoch=epoch, **kwargs)

                if show and i % show_frequency == 0:
                    Y_hat = self._feed_forward(X)
                    J_list[epoch] = self.LOSS(Y, Y_hat) \
                                  + self.lambda1 * np.sum(np.sum(abs(W)) for W in self.W) \
                                  + self.lambda2/2 * np.sum(np.sum(W**2) for W in self.W)

        if show:
            plt.figure()
            plt.plot(J_list)
            plt.show()

    def predict(self, X):
        return self._feed_forward(X)

    def predict_category(self, X):
        return np.argmax(self.predict(X), axis=1)

    @staticmethod
    def _one_hot_encode(y, K):
        N = len(y)
        Y = np.zeros((N, K))
        for i in range(N):
            Y[i, y[i]] = 1
        return Y

    def _L(self):
        """
        :return: ID of final layer in network.
        """
        return len(self.W)

    def _layers(self):
        """
        :return: Enumeration of layers in network.
        """
        return range(1, self._L()+1)

    def _feed_forward(self, X):
        self.Z[0] = X
        for l in self._layers():
            self.Z[l] = self.A[l].function(np.matmul(self.Z[l-1], self.W[l]) + self.b[l])
        return self.Z[self._L()]

    def _back_propagate(self, Y, eta, **kwargs):
        t = kwargs["epoch"]

        for l in self._layers()[::-1]:
            # dH for output layer.
            if l == self._L():
                # Z[l] is Y_hat for regression and P_hat for classification.
                self.dH[l] = self.Z[l] - Y

            # dH for other layers.
            else:
                self.dZ[l] = np.matmul(self.dH[l+1], self.W[l+1].T)
                self.dH[l] = self.dZ[l] * self.A[l].derivative(self.Z[l])

            self.dW[l] = np.matmul(self.Z[l-1].T, self.dH[l])
            self.db[l] = self.dH[l].sum(axis=0)
            self.W[l] = self.OPTIMIZER_W.update(W=self.W[l], dW=self.dW[l], eta=eta, l=l, t=t) \
                      - (self.lambda1 * np.sign(self.W[l]) + self.lambda2 * self.W[l])
            self.b[l] = self.OPTIMIZER_b.update(W=self.b[l], dW=self.db[l], eta=eta, l=l, t=t)

    def add(self, units, activation, input_dimension=None):
        L = self._L()

        # Get the input dimension for the new layer.
        if self._L() == 0:
            try:
                assert input_dimension is not None
            except AssertionError:
                raise ValueError("First layer in network must specify the input dimension D.")
            M_in = input_dimension
        else:
            M_in = self.W[L].shape[1]

        # The output dimension for the new layer.
        M_out = units

        # Layer will be added to last layer in network.
        L = self._L() + 1

        # Create the weight matrices for the new layer.
        self.W[L] = np.random.randn(M_in, M_out)
        self.b[L] = np.random.randn(M_out)

        # Add the new activation function.
        self.A[L] = activation

    def _N(self):
        return self.Z[0].shape[0]

    def _D(self):
        return self.Z[0].shape[1]

    def _K(self):
        return self.Z[self._L()].shape[1]

    @staticmethod
    def _cross_entropy(Y, P_hat):
        return -np.sum(Y * np.log(P_hat))

    @staticmethod
    def _OLS(Y, Y_hat):
        return np.sum(np.matmul((Y - Y_hat).T, (Y - Y_hat)))

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
        layers = ", ".join(f'(units={self.W[l].shape[1]}, {self.A[l]})' for l in self._layers())
        return f"FeedForwardNeuralNetwork(loss={self.loss}, lambda1={self.lambda1}, lambda2={self.lambda2}, layers={layers})"
