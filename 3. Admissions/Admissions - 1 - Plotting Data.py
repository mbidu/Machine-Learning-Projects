# Shift + Alt + A

import sys

# used for manipulating directory paths
import os

import matplotlib.pyplot as plt

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Scientific and vector computation for python
import numpy as np

# Load data
# The first two columns contains the exam scores and the third column
# contains the label.

pathdata = 'Exam Scores, Admissions.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0:2], data[:, 2]

def plotData(X, y):

    # Create New Figure
    fig = pyplot.figure()

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'gP', mec='k', mew=1, ms=8)
    pyplot.plot(X[neg, 0], X[neg, 1], 'rX', mec='k', mew=1, ms=8,)
    pyplot.title('Admissions Based on Exam Scores')
    pyplot.xlabel('Exam 1 Score (%)')
    pyplot.ylabel('Exam 2 Score (%)')
    pyplot.legend(['Admitted', 'Not Admitted'])

plotData(X, y)
plt.show()

m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)