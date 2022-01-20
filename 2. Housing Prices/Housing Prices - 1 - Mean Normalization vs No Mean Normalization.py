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

pathdata = 'Square Feet, Bedrooms, Housing Prices.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')

X = data[:, :2]
y = data[:, 2]
m = y.size

# print out some data points
print('\n{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

def  MeanNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

X_norm, mu, sigma = MeanNormalize(X)
X_orig = X.copy()

print('\nComputed mean:', mu)
print('Computed standard deviation:', sigma, '\n')

XNorm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
XOrig = np.concatenate([np.ones((m, 1)), X_orig], axis=1)

def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0

    h = np.dot(X, theta)
    J = (1/(2*m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

alpha = 0.1
num_iters = 400

thetaNorm = np.zeros(3)
theta, J_history = gradientDescentMulti(XNorm, y, thetaNorm, alpha, num_iters)
thetaNorm = theta
J_historyNorm = J_history

J_history = []
thetaOrig = np.zeros(3)
theta, J_history = gradientDescentMulti(XOrig, y, thetaOrig, alpha, num_iters)
thetaOrig = theta
J_historyOrig = J_history

fig = pyplot.figure(figsize=(12, 5))

ax = pyplot.subplot(121)
pyplot.plot(np.arange(len(J_historyNorm)), J_historyNorm, lw=2, label = "Mean Normalized")
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.title('Food Truck Profits and Population Density\nGradient Decent + Mean Normalized')

ax = pyplot.subplot(122)
pyplot.plot(np.arange(len(J_historyOrig)), J_historyOrig, lw=2, label = "Original")
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.title('Food Truck Profits and Population Density\nGradient Decent (Overflow)')
pass

plt.show()

print('theta computed from gradient descent: {:s}'.format(str(thetaOrig)))
print('theta computed from gradient descent with Mean Normalization: {:s}'.format(str(thetaNorm)))

X_array = [1, 1650, 3]
priceOrig = np.dot(X_array, thetaOrig)

X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma
priceNorm = np.dot(X_array, thetaNorm)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(priceOrig))
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent with Mean Normalization): ${:.0f}'.format(priceNorm))

# Notice that there is overflow when you do not use Mean Normalization or Feature Scaling