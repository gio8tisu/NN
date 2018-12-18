"""Module with utils for neural networks and logistic regression."""

import numpy as np


def identity(z):
    return z


def sigmoid(z):
    """return sigmoid function of z."""
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    """return derivative of sigmoid at point z."""
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1 - tanh(z) ** 2


def relu(z):
    return np.maximum(z, 0)


def d_relu(z):
    return np.heaviside(z, 1)


def softmax(z):
    z_exp = np.exp(z - np.max(z))  # shift to stabilize
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)


def loss(a, y):
    """return cross-entropy loss of a and y."""
    if y == 1:
        return -np.log(a)
    else:
        return -np.log(1 - a)


def cost(a, y):
    """return cost, i.e. mean of loss over each sample."""
    return np.mean(list(map(loss, a, y)))
