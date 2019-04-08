import matplotlib.pyplot as plt
import numpy as np

from TFTL import RegressionModel


class LinearRegressionModel(RegressionModel.RegressionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Weights of the linear model
        self.W = None

    def fit(self, X, Y, method="analytic", **kwargs):
        if method == "gradient":
            self.fit_with_gradient_descent(X, Y, **kwargs)
        if method == "analytic":
            self.fit_analytic(X, Y)
        else:
            raise NotImplemented(f"Fit method \"{method}\" not implemented.")

    def predict(self, X):
        return self.phi(X) @ self.W

    def J(self, X, Y):
        n = X.shape[0]
        L1_regularization = self.lambda1 * abs(self._I_reg() @ self.W).sum()
        L2_regularization = self.lambda2/2 * (self.W.T @ self._I_reg() @ self.W).sum()
        return self.MSE(X, Y) + L1_regularization/n + L2_regularization/n

    def fit_with_gradient_descent(self, X, Y, learning_rate, epochs, plot=False):
        self._format(X, Y, train=True)

        J_list = []

        self.W = np.random.randn(self.D(), self.Y.shape[1])
        #self.W = np.zeros((self.design_feature_count(), self.Y.shape[1]))

        print(self.PHI)
        print(self.W)
        print(self.PHI @ self.W)

        for epoch in range(int(epochs)):
            self.Y_hat = self.PHI @ self.W
            J_list.append(self.J(X, Y))
            self.W -= learning_rate * (
                    self.PHI.T @ (self.Y_hat - self.Y)
                    + self.lambda1 * self._I_reg() @ np.sign(self.W)  # L1 regularization gradient
                    + self.lambda2 * self._I_reg() @ self.W  # L2 regularization gradient
            )

        self.Y_hat = self.predict(self.X)

        if plot:
            plt.figure()
            plt.plot(J_list)
            plt.show()

    def fit_analytic(self, X, Y, **kwargs):
        """
        Normal Equation solution.
        :param X:
        :param Y:
        :return:
        """
        self.X = X
        self.Y = Y
        self.PHI = self.phi(X)

        self.W = np.linalg.solve(self.PHI.T @ self.PHI, self.PHI.T @ Y)

        self.Y_hat = self.predict(X)

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
