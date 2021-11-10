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

path = os.path.join('/Users/mackt/Python/Machine Learning/Data', 'Square Feet, Housing Prices.txt')
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size
X = np.stack([np.ones(m), X], axis=1)

#Computing J for given theta
def computeCost(X, y, theta):
    J = 0

    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    theta = theta.copy()
    J_history = [] # Use a python list to save cost in every iteration

    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

theta = np.zeros(2)
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('\nTheta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]\n')

predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}'.format(predict1*10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

def plotData(x, y):
    fig = plt.figure()
    plt.plot(X[:, 1], y, 'go', ms=7, mec='k')
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression']);
    plt.ylabel('Food Truck Profits ($10,000s)')
    plt.xlabel('Population of City in (10,000s)')
    pyplot.title('Food Truck Profits and Population Density')
plotData(X, y)

plt.show()