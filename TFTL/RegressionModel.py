import matplotlib.pyplot as plt
import numpy as np

from TFTL.SupervisedModel import SupervisedModel


class RegressionModel(SupervisedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Independent variables
        self.X = None

        # Dependent variables
        self.Y = None

        # Data transformation
        if "phi" not in kwargs:
            self.phi = lambda X: X
        else:
            self.phi = kwargs["phi"]

        # Predicted independent variables
        self.Y_hat = None

        # Design matrix
        self.PHI = None

        # L1 norm regularization factor
        self.lambda1 = kwargs["lambda1"] if "lambda1" in kwargs else 0

        # L2 norm regularization factor
        self.lambda2 = kwargs["lambda2"] if "lambda2" in kwargs else 0

        # TODO: normalize the data


    def fit(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.PHI = self.phi(X)

        # Find model parameters based on training data.
        raise NotImplemented("Regression Model must be inherited from and fit() implemented.")

        # Find the values of y_hat.
        raise NotImplemented("fit() must be implemented to find y_hat.")

    def predict(self, Z):
        PHI_Z = self.phi(Z)

        raise NotImplemented("Regression Model must be inherited from and predict() implemented.")

    def J(self, X, Y):
        """
        Objective function
        :return:
        """
        return self.MSE(X, Y)

    def MSE(self, X, Y):

        n = X.shape[0]
        # Ordinary least squares error.
        Y_hat = self.predict(X)

        #print(list(zip(Y[:10], Y_hat[:10])))

        # Compute the Frobenius.
        mse = np.linalg.norm((Y - Y_hat), ord='fro')**2 / n
        return mse

    def r_squared(self, X=None, Y=None, combined=True):
        """
        :combined: Whether R^2 should be combined or not for multivariate regression.
        :return: R^2 value of the model on the data.
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        Y_hat = self.predict(X)

        def _r_squared(_Y_hat, _Y):
            return 1 - np.sum((_Y - _Y_hat) ** 2) / np.sum((_Y - _Y.mean()) ** 2)

        if combined:
            return _r_squared(Y_hat, Y)
        else:
            return [_r_squared(y_hat, y) for y_hat, y in zip(Y_hat.T, Y.T)]

    def plot_fit(self, color="#5500FF"):
        # Look at the fit.
        plt.figure()
        plt.scatter(self.X, self.Y)
        plt.plot(self.X, self.Y_hat, color, linewidth=2)
        plt.show()

    def plot_residuals(self):
        """
        Create a plot to check that the residuals don't form a pattern.
        :return: None
        """
        residuals = self.Y_hat - self.Y
        plt.figure()
        plt.scatter(self.X, residuals)
        plt.show()
