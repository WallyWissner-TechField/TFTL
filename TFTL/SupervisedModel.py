import numpy as np
import pandas as pd


from TFTL.Model import Model

class SupervisedModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Y = None

    def _normalize(self, PHI, PHI_train=None):
        """
        Normalize a matrix to between min and max of another matrix.
        :param PHI: Matrix to normalize.
        :param PHI_train: Matrix whose values should be normalized against.
        :return: Normalized matrix.
        """
        PHI_train = PHI_train if PHI_train.any() else PHI
        mins = [min(col) for col in PHI_train.T]
        maxs = [max(col) for col in PHI_train.T]
        normalized = np.array((col - mini)/(maxi - mini) for mini, maxi, col in zip(mins, maxs, PHI.T)).T
        return normalized

    def _format(self, X, Y=None, train=False, normalize=True):
        """
        :pd.df or np.array X: Data.
        :bool train:  Whether the data is training data.
        :return:
        """
        # Make Y a column vector matrix instead of an array.
        if Y is not None:
            try:
                Y = np.array(Y)
                Y.reshape(Y.shape[0], Y.shape[1])
            except IndexError:
                Y.reshape(Y.shape[0], 1)

        PHI = self.phi(X)
        PHI = np.array(PHI)
        PHI_train = PHI if train else self.PHI
        if normalize:
            PHI = self._normalize(PHI, PHI_train=PHI_train)
        if train:
            self.X = X
            if Y is not None:
                self.Y = Y
            self.PHI = PHI
        return PHI, Y

    def D(self):
        print(self.PHI)
        return self.PHI.shape[0]

