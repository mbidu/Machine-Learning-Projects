# Shift + Alt + A

import sys
import os

import matplotlib.pyplot as plt

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Scientific and vector computation for python
import numpy as np

from scipy import optimize

scriptpath = "C:/Users/mackt/Python/Machine Learning/3. Microchips/utils"
sys.path.append(os.path.abspath(scriptpath))

# Load data
# The first two columns contains the exam scores and the third column contains the label.
pathdata = 'Microchip Tests, Pass QA.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0:2], data[:, 2:3]

def plotdata(X, y):
    postive = np.where(y > 0.5)
    negtive = np.where(y < 0.5)
    plt.scatter(X[postive[0],0],X[postive[0],1],marker='+',c='g')
    plt.scatter(X[negtive[0],0],X[negtive[0],1],marker='x',c='r')
    pyplot.title('QA Testing Results')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    pyplot.legend(['Pass', 'Fail'], loc='upper right')

# Plot Data
plotdata(X, y)
plt.show()