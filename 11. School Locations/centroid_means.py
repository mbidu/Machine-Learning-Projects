import numpy as np

def centroid_means(X, idx, K):

    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        x = X[idx == k]
        centroids[k, :] = np.mean(x, axis=0)

    return centroids

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