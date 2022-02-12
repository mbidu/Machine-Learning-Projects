import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
from sklearn import svm

from plot_data import plot_data
from visualize_boundary_linear import visualize_boundary_linear

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Loading and Visualizing Data ========================
    print('='*10, 'Part 1: Loading and Visualizing Data', '='*10)
    print("\nLoading Data...\n")

    pathdata = '7. Recognition Software.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    X = data['X']
    y = data['y'].ravel()

    print("Plotting Data...\n")
    plt.figure()
    plot_data(X,y)
    plt.xlabel('Correct Dogs Found (x100)')
    plt.ylabel('Correct Humans Found (x100)')
    plt.title('Recognition Software Test')
    plt.xlim([0, 4.5])
    plt.ylim([1.5, 5])
    plt.show()

    # ========================= Part 2: Training Linear Kernel SVM ========================
    print('='*10, 'Part 2: Training Linear Kernel SVM', '='*10)
    print("\nTraining SVM...\n")

    # C = 1/Lambda
    for C in [0.01,1,100]:
        clf = svm.LinearSVC(C=C, dual=False)
        clf.fit(X, y)
        acc = clf.score(X, y)*100
        print('C = %.2f\tAccuracy: %.2f' %(C, acc),'%')

        plt.figure()
        visualize_boundary_linear(X, y, clf)
        plt.xlabel('Correct Dogs Found (x100)')
        plt.ylabel('Correct Humans Found (x100)')
        plt.title('Recognition Software Test C = %.2f' %C)
        plt.xlim([0, 4.5])
        plt.ylim([1.5, 5])
        plt.show()

    print('\n','='*22, "End", '='*22)