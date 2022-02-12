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

scriptpath = "C:/Users/mackt/Python/Machine Learning/2. Admissions/utils"
sys.path.append(os.path.abspath(scriptpath))

import utils

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

def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

# Setup the data matrix appropriately
m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta.T))
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))
    grad = (1 / m) * (h - y).dot(X)
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

print('\nCost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):[0.043, 2.566, 2.647]\n')

# set options for optimize.minimize
options= {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
res = op.minimize(costFunction, initial_theta, (X, y), jac=True, method='TNC', options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):[-25.161, 0.206, 0.201]\n')

utils.plotDecisionBoundary(plotData, theta, X, y)
plt.show()

def predict(theta, X):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)
    p = np.round(sigmoid(X.dot(theta.T)))
    return p

prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
        ' we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %\n')