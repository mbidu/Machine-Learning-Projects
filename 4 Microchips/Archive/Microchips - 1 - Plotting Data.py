# Shift + Alt + A

import sys

# used for manipulating directory paths
import os

import matplotlib.pyplot as plt

import scipy.optimize as op

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Scientific and vector computation for python
import numpy as np

scriptpath = "C:/Users/mackt/Python/Machine Learning/3. Microchips/utils"
sys.path.append(os.path.abspath(scriptpath))

# Load data
# The first two columns contains the exam scores and the third column
# contains the label.
pathdata = 'Microchip Tests, Pass QA.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0:2], data[:, 2:3]

def plotData(X, y):

    # Create New Figure
    fig = pyplot.figure()

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'gP', mec='k', mew=1, ms=8)
    pyplot.plot(X[neg, 0], X[neg, 1], 'rX', mec='k', mew=1, ms=8)
    pyplot.title('QA Testing Results')
    pyplot.xlabel('Microchip Test 1')
    pyplot.ylabel('Microchip Test 2')
    pyplot.legend(['Pass', 'Fail'], loc='upper right')

plotData(X,y)
plt.show()