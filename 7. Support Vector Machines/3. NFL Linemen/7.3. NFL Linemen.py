import os
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
from sklearn import svm

from plot_data import plot_data
from NFL_Linemen_params import NFL_Linemen_params
from visualize_boundary import visualize_boundary

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Loading and Visualizing Data ========================
    print('='*10, 'Part 1: Loading and Visualizing Data', '='*10)
    print("\nLoading Data...\n")

    pathdata = '7.3. NFL Linemen.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    X = data['X']
    y = data['y'].ravel()
    X_val = data['Xval']
    y_val = data['yval'].ravel()

    # Plot training data
    plt.figure()
    plot_data(X, y)
    plt.xlabel('40-Yard Dash')
    plt.ylabel('Bench Press')
    plt.title('NFL Linemen')
    plt.show()

    # ========================= Part 2: Training RBF Kernel SVM ========================
    print('='*10, 'Part 2: Training RBF Kernel SVM', '='*10)
    print("\nLoading Data...\n")

    # Try different SVM Parameters here
    # gamma is the kernel normalization coefficient
    # gamma = -1 / (2*standard_deviation^2) = -1 / (2*variance)
    # k(x1, x2) = e^(gamma||x1 - x2||^2)
    # C = 1 / Lambda

    C, gamma = NFL_Linemen_params(X, y, X_val, y_val)

    # Train the SVM
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(X, y)
    acc = clf.score(X, y)*100
    print('C = %i\t\tgamma = %i\tAccuracy: %.2f' %(C, gamma, acc))

    plt.figure()
    warnings.simplefilter("ignore", UserWarning)
    visualize_boundary(X, y, clf)
    plt.xlabel('40-Yard Dash')
    plt.ylabel('Bench Press')
    plt.title('NFL Linemen')
    plt.show()

    print('\n','='*22, "End", '='*22)