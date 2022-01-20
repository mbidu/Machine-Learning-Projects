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

path = os.path.join('/Users/mackt/Python/Machine Learning/Data', 'Food Truck Profits ($10,000), Population (10,000s).txt')
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size

def plotData(x, y):
    fig = plt.figure()
    plt.plot(x, y, 'go', ms=7, mec='k')
    plt.ylabel('Food Truck Profits ($10,000s)')
    plt.xlabel('Population of City in (10,000s)')
    pyplot.title('Food Truck Profits and Population Density')

plotData(X, y)
plt.show()