"""Module with loss functions."""

import numpy as np


def loss(a, y):
    """return cross-entropy loss of a and y."""
    if y == 1:
        return -np.log(a)
    else:
        return -np.log(1 - a)


def cost(a, y):
    """return cost, i.e. mean of loss over each sample."""
    return np.mean(list(map(loss, a, y)))
