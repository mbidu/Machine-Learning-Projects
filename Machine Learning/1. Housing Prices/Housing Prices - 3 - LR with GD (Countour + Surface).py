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

pathdata = 'Square Feet, Housing Prices.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
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

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('\nTheta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]\n')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}'.format(predict1*10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# Create Figure
fig = pyplot.figure(figsize=(15, 5))

# Plot Data With Regression
ax = pyplot.subplot(131)
plt.plot(X[:, 1], y, 'go', ms=7, mec='k')
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);
plt.ylabel('Food Truck Profits ($10,000s)')
plt.xlabel('Population of City in (10,000s)')
pyplot.title('Food Truck Profits and Population Density')

# Surface Plot
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface Plot')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(133)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'go', ms=10, lw=2)
pyplot.title('Contour Plot, Showing Minimum J')
pass

plt.show()