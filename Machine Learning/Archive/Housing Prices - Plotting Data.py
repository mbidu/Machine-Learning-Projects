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

path = os.path.join('/Users/mackt/Python/Machine Learning/Data', 'ex1data1.txt')

print(path)

data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size

print(m)

def plotData(x, y):
    fig = plt.figure()
    plt.plot(x, y, 'go', ms=7, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
plotData(X, y)

plt.show()

X = np.stack([np.ones(m), X], axis=1)

#Computing J for given theta
def computeCost(X, y, theta):
    J = 0
    
    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

J = computeCost(X, y, theta=np.array([0.0, 0.0]))
J1 = J
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

J = computeCost(X, y, theta=np.array([-1, 2]))
J2 = J
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24\n')

if J1 < J2:
    print('J1 is closer than J2 to minimizing J.\n')
elif J1 > J2:
    print('J2 is closer than J1 to minimizing J.\n')
else:
    print('J1 and J2 are equally close to minimizing J.\n')

def gradientDescent(X, y, theta, alpha, num_iters):
    theta = theta.copy()

    J_history = [] # Use a python list to save cost in every iteration

    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)

        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# plot the linear fit
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);
plt.show()