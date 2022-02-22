import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from plot_data_points import plot_data_points
from connect_data_points import connect_data_points

def plot_k_means_progress(X, history_centroids, idx, K, i):

    plot_data_points(X, idx, K)
    for k in range(K):
        c = cm.rainbow(np.linspace(0, 1, K))
        plt.plot(history_centroids[0:i+1, k:k+1, 0], history_centroids[0:i+1, k:k+1, 1], linestyle='', marker='X', markersize=10, linewidth=3, color=c[k], mec = 'k')
    plt.title('School Locations - Iteration {0}'.format(i + 1))
    for centroid_idx in range(history_centroids.shape[1]):
        for iter_idx in range(i):
            connect_data_points(history_centroids[iter_idx, centroid_idx, :], history_centroids[iter_idx + 1, centroid_idx, :])


    """
    Returns the new centroids by computing the means of the data points assigned to each centroid.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    idx : ndarray, shape(n_samples, 1)
        Centroid assignments.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
        New centroids, each row of which is the mean of the data points assigned to it.

    """