import numpy as np
import matplotlib.pyplot as plt

from closest_centroids import closest_centroids
from centroid_means import centroid_means
from plot_k_means_progress import plot_k_means_progress

def k_means(X, initial_centroids, max_iters, plot_progress=False):

    K = initial_centroids.shape[0]
    centroids = initial_centroids
    history_centroids = np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
    idx = np.zeros(X.shape[0])

    for i in range(max_iters):
        print('K-Means iteration {0}/{1}'.format(i + 1, max_iters))
        history_centroids[i, :] = centroids

        idx = closest_centroids(X, centroids)

        if plot_progress:
            plt.figure()
            plot_k_means_progress(X, history_centroids, idx, K, i)
            plt.show()

        centroids = centroid_means(X, idx, K)

    return centroids, idx

    """

    Runs the K-Means algorithm on data matrix X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    initial_centroids : ndarray, shape (K, n_features)
        The initial centroids.
    max_iters : int
        Total number of iteration for K-Means to execute.
    plot_progress : bool
        True to plot progress for each iteration.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
        The final centroids.
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments. idx[i] contains the index of the centroid closest to sample i.

    """