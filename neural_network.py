import numpy as np

import utils


class Layer:
    def __init__(self, d_input, n_units, activation="identity"):
        self.d_input = d_input
        self.n_units = n_units
        self.W = np.random.randn(n_units, d_input)
        self.b = np.zeros((n_units, 1))
        try:
            self.activation = getattr(utils, activation)
        except AttributeError:
            raise AttributeError("Activation function doesn't exist.")

    def forward(self, a, y=None):
        assert a.shape[0] == self.W.shape[1]
        z = np.dot(self.W, a) + self.b
        return self.activation(z)

    def backward(self, X, y, a):
        pass
