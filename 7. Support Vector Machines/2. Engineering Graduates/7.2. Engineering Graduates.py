import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
from sklearn import svm

from gaussian_kernel import gaussian_kernel
from plot_data import plot_data
from visualize_boundary import visualize_boundary

import warnings

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Implement Gaussian Kernel ========================
    print('='*10, 'Part 1: Implement Gaussian Kernel', '='*10)
    print("\nEvaluating Gaussian Kernel...\n")

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %i: \t%.2f' %(sigma,sim))
    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2: \t0.32 (Expected)\n')

    # ========================= Part 2: Loading and Visualizing Data ========================
    print('='*10, 'Part 2: Loading and Visualizing Data', '='*10)
    print("\nLoading Data...\n")

    pathdata = '7.2. Engineering Graduates.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    X = data['X']
    y = data['y'].ravel()

    # Plot training data
    plt.figure()
    plot_data(X, y)
    plt.xlabel('Grade 9 Gym Mark')
    plt.ylabel('Grade 9 English Mark')
    plt.title('Engineering Graduates')
    plt.show()

    # ========================= Part 3: Training RBF Kernel SVM ========================
    print('='*10, 'Part 3: Training RBF Kernel SVM', '='*10)
    print("\nLoading Data...\n")

    # SVM Parameters
    C = 100     # C = 1/Lambda
    gamma = 10

    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(X, y)
    acc = clf.score(X, y)*100
    print('C = %i\t\tgamma = %i\tAccuracy: %.2f' %(C, gamma, acc))

    plt.figure()
    warnings.simplefilter("ignore", UserWarning)
    visualize_boundary(X, y, clf)
    plt.xlim([0, 1])
    plt.ylim([0.4, 1])
    plt.xlabel('Grade 9 Gym Mark')
    plt.ylabel('Grade 9 English Mark')
    plt.title('Engineering Graduates')
    plt.show()

    print('\n','='*22, "End", '='*22)