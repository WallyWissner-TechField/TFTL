import abc
import numpy as np

from TFTL.SupervisedModel import SupervisedModel


class ClassifierModel(SupervisedModel):
    def __init__(self, *args, **kwargs):
        self.__meta__ = abc.ABCMeta
        super().__init__(*args, **kwargs)
        self.X = None
        self.Y = None

    @abc.abstractmethod
    def predict(self, X):
        return

    def sigmoid(self, h):
        return 1 / (1 + np.exp(-h))

    def J(self, Y, P_hat):
        """
        Cross entropy error.
        :param Y:
        :param P_hat:
        :return:
        """
        return -np.sum(Y * np.log(P_hat))

    def accuracy(self, X=None, Y=None):
        X = X if X else self.X
        Y = Y if Y else self.Y
        P_hat = self.predict(X)
        return np.mean(Y == np.round(P_hat))
