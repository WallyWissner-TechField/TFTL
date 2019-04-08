import abc


class ActivationFunction(object):
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def function(self, H):
        """
        Activation function.
        :param H: Logit
        :return:
        """
        return

    @abc.abstractmethod
    def derivative(self, H):
        """
        Derivative of the activation function.
        :param H: Logit
        :return:
        """
        return

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __repr__(self):
        return self.__str__()
