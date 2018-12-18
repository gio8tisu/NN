"""Tools for logistic regression."""

import time

import numpy as np
import matplotlib.pyplot as plt

from utils import cost, loss, sigmoid, relu

class Model:
    """Class implementing logistic regression.

    """

    def __init__(self, optimizer='gd', learning_rate=0.001, verbose=False, plot=False):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = np.inf
        self.verbose = verbose
        self.plot = plot

    @property
    def weights(self):
        return self.w

    def fit(self, X, y, batch_size=None, max_iter=100):
        """Fit logistic regression model with data X.

        :arg X - numpy matrix with data observations as rows
        :arg y - numpy array with data labels

        :return trained model
        """
        if not batch_size:
            batch_size = X.shape[0]

        # Initialize weights and bias.
        self.w = np.random.randn(X.shape[1], 1)
        self.b = 0
        i = 0
        if self.plot:
            # Plot data and decision boundary.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.xlim = (-10, 10)
            ax.ylim = (-10, 10)
            c = ["r", "b"]
            colors = [c[y_i] for y_i in y]
            ax.scatter(X[:, 0], X[:, 1], c=colors)
            line, = ax.plot([-10, 10],
                            [10 * self.w[0] / self.w[1] - self.b / self.w[1],
                             -10 * self.w[0] / self.w[1] - self.b / self.w[1]],
                            c="black")
            plt.show()

        if self.verbose:
            tic = time.time()
        # Actual training, each while-loop is an iteration/epoch.
        while i < max_iter and self.loss > 0.000001:
            for k in range(0, X.shape[0] // batch_size):
                # Use a subset of training data to update parameters.
                rang = range(k*batch_size, (k + 1) * batch_size)
                a = self.forward(X[rang, ], y[rang])
                if self.verbose:
                    print(i, k, "Loss:", self.loss)
                self.backward(X[rang, ], y[rang], a)
            i += 1
            if self.plot and i % 100 == 0:
                # Re-plot boundary with updated parameters.
                line.set_ydata([10 * self.w[0] / self.w[1] - self.b / self.w[1],
                                -10 * self.w[0] / self.w[1] - self.b / self.w[1]])
                fig.canvas.show()

        if self.verbose:
            toc = time.time()
            print("Accuracy: ", 100 * np.mean(y == np.round(self.predict(X))), "%")
            print("Elapsed training time: ", 1000 * (toc - tic), "ms")
        return self

    def forward(self, X, y):
        """Forward pass.

        first, compute dot product of all samples X with weights self.w
        pass the result through sigmoid function, i.e get probability prediction
        :return the loss between the predictions and the labels
        """
        a = self.predict(X)
        self.loss = cost(a, y)
        return a

    def backward(self, X, y, a):
        """Backward pass.

        Compute derivatives and update weights and bias
        """
        dz = (a - y) / X.shape[0]
        dw = np.dot(X.T, a - y)

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * sum(dz)

    def predict(self, X):
        """Predict probability of data being positive."""
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
