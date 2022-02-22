import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat

from closest_centroids import closest_centroids
from centroid_means import centroid_means
from k_means import k_means

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Find Closest Centroid ========================
    print('='*10, 'Part 1: Find Closest Centroid', '='*10)
    print('\nFinding Closest Centroids...\n')

    # Load 300 2D Coordinates
    pathdata = '11. House_Locations.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    # Housing Locations
    X = data['X'] # (300, 2)

    # Select an initial set of centroids
    K = 3  # Number of Centroids (Number of Schools)
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Find the closest centroid to each 2D coordinate
    idx = closest_centroids(X, initial_centroids)

    print('Closest centroids for the first 3 examples:', idx[0:3])
    print('Closest centroids for the first 3 examples: [0, 2, 1] (Expected)\n')

    # ========================= Part 2: Centroids Means ========================
    print('='*10, 'Part 2: Centroids Means', '='*10)
    print('\nComputing Centroids Means...\n')

    # Compute means based on the closest centroids found in the previous part.
    # Each 2D coordinate was assigned to one of the K centroids
    # The mean of each 2D coordinate assigned to the same centroid is calculated as the new centroid
    centroids = centroid_means(X, idx, K)
    centroids = np.round(centroids, decimals = 6)

    print('Centroids computed after finding of closest centroids:', centroids[0],centroids[1],centroids[2])
    print('Centroids computed after finding of closest centroids: [2.428301 3.157924] [5.813503 2.633656] [7.119387 3.616684] (Expected)\n')

    # ========================= Part 3: K-Means Clustering ========================
    print('='*10, 'Part 3: K-Means Clustering', '='*10)
    print('\nRunning K-Means Clustering...\n')

    # Load 300 2D Coordinates
    pathdata = '11. House_Locations.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    X = data['X'] # (300, 2)

    # Settings for running K-Means
    K = 3
    max_iters = 10

    # For consistency, here we set centroids to specific values but in practice you want to generate them automatically,
    # such as by settings them to be random examples (as can be seen in k_means_init_centroids).
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Run K-Means algorithm
    centroids, idx = k_means(X, initial_centroids, max_iters, True)

    print('\n','='*22, "End", '='*22)