import numpy as np
import matplotlib.pyplot as plt

import logistic_regression

import logistic_regression


def main(s=10000, d=1000, l=0.01, iterations=1000):
    X = np.random.randn(s, d) - 1
    X = 5 * np.vstack((X, np.random.randn(s, d) + 1))
    y = np.array([0] * s + [1] * s).reshape((X.shape[0], 1))
    model = logistic_regression.Model(learning_rate=l,
                                      verbose=True, plot=True)
    model.fit(X, y, max_iter=iterations, batch_size=516)
    X_test = np.random.randn(10, d)
    y_pred = model.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], c="g")
    plt.draw()
    print(y_pred)
    print(model.weights)


if __name__ == "__main__":
    main()
