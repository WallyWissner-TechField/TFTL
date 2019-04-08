import matplotlib.pyplot as plt
import numpy as np

from TFTL.ClassifierModel import ClassifierModel


class LinearClassifierModel(ClassifierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # L1 norm regularization factor
        self.lambda1 = kwargs["lambda1"] if "lambda1" in kwargs else 0
        # L2 norm regularization factor
        self.lambda2 = kwargs["lambda2"] if "lambda2" in kwargs else 0

        # Weights of the linear model
        self.W = None

    def fit(self, X, Y, learning_rate, epochs, plot=False):
        self._format(X, Y, train=True)

        self.PHI = self._stack_ones(self.PHI)

        self.W = np.random.randn(self.D(), self.Y.shape[1])

        Js = [0] * epochs

        for epoch in range(epochs):
            p_hat = self.predict(self.PHI)
            Js[epoch] = self.J(Y, p_hat)
            self.W -= learning_rate * (
                np.matmul(self.PHI.T, p_hat - Y)
                + self.lambda1 * self._I_reg() @ np.sign(self.W)  # L1 regularization gradient
                + self.lambda2 * self._I_reg() @ self.W  # L2 regularization gradient
                )

        if plot:
            plt.figure()
            plt.plot(Js)
            plt.show()

    def predict(self, X):
        PHI = self._format(X, train=False)
        return self.sigmoid(np.matmul(PHI, self.W))

    def predict_category(self, X):
        return np.argmax(self.predict(X), axis=1)

    def J(self, Y, P_hat):
        """
        Cross entropy error with regularization.
        :param Y:
        :param P_hat:
        :return:
        """
        return -np.sum(Y * np.log(P_hat)) \
                + self.lambda1 * np.sum(abs(self.W)) \
                + self.lambda2/2 * np.sum(self.W**2)

    def _I_reg(self, N=None):
        """
        Create the I_reg matrix of size NxN, the identity matrix of size NxN but with the first element 0.
        Used for regularization. First element is zero so bias weight is the average of the data.
        :param N: Size of matrix.
        :return: I_reg matrix of size NxN.
        """
        if N is None:
            N = self.D()
        i_reg = np.identity(N)
        i_reg[0, 0] = 0
        return i_reg
