import numpy as np
from matplotlib import pyplot as plt


def plot_data(X, y):
    pos = np.nonzero(y == 1)
    neg = np.nonzero(y == 0)

    plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='*', color='g')
    plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='x', color='r')

    """
    Plots the data points X and y into a new figure.
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Target values (class labels in classification).
    """