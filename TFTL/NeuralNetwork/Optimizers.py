import numpy as np

from collections import defaultdict


class Adam(object):
    def __init__(self, eta, gamma=.999, mu=.9, epsilon=1e-8, **kwargs):
        self.eta = eta  # Learning rate.
        self.m = defaultdict(float)  # Mean.
        self.v = defaultdict(float)  # Variance.
        self.gamma = gamma  # Memory.
        self.mu = mu  # Friction.
        self.epsilon = epsilon  # Offset.
        self.t = 1  # Epoch.

    def update(self, W, dW, l, t, **kwargs):
        t += 1
        self.m[l] = self.mu * self.m[l] + (1 - self.mu) * dW
        self.v[l] = self.gamma * self.v[l] + (1 - self.gamma) * dW ** 2
        m_hat = self.m[l] / (1 - self.mu ** self.t)
        v_hat = self.v[l] / (1 - self.gamma ** self.t)
        W -= self.eta / np.sqrt(v_hat + self.epsilon) * m_hat
        return W


class RMSPropMomentum(object):
    def __init__(self, eta, gamma=.999, mu=.9, epsilon=1e-8, **kwargs):
        self.eta = eta  # Learning rate.
        self.G = defaultdict(float)  # Cache.
        self.v = defaultdict(float)  # Momentum.
        self.gamma = gamma  # Memory.
        self.mu = mu  # Friction.
        self.epsilon = epsilon  # Offset.

    def update(self, W, dW, l, **kwargs):
        self.G[l] = self.gamma * self.G[l] + (1 - self.gamma) * dW ** 2
        self.v[l] = self.mu * self.v[l] - self.eta * dW
        W += self.eta / np.sqrt(self.G[l] + self.epsilon) * self.v[l]
        return W


class StochasticGradientDescent(object):
    def __init__(self, eta, **kwargs):
        self.eta = eta  # Learning rate.

    def update(self, W, dW, **kwargs):
        W -= self.eta * dW
        return W
