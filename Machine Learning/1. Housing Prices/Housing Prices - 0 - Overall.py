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

def plotData(x, y):
    plt.plot(x, y, 'go', ms=7, mec='k')
    plt.ylabel('Food Truck Profits ($10,000s)')
    plt.xlabel('Population of City in (10,000s)')
    pyplot.title('Food Truck Profits and Population Density')

fig = plt.figure()
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
print('\nWith theta = [0, 0] \nCost computed = %.2f' % J)
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
print('Expected theta values (approximately): [-3.6303, 1.1664]\n')

# plot the linear fit
def LinearFit(X, y, theta):
    plotData(X[:, 1], y)
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression']);

LinearFit(X, y, theta)
plt.show()

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
fig = pyplot.figure(figsize=(12, 5))

# Plot Data With Regression
ax = fig.add_subplot(131)
LinearFit(X, y, theta)

# surface plot
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
pyplot.title('Contour Plot Showing Minimum')
pass

plt.show()

#LINEAR REGRESSION WITH MULTIPLE VARIABLES

pathdata = 'Square Feet, Bedrooms, Housing Prices.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
print('\n')

def  MeanNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

X_norm, mu, sigma = MeanNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma, '\n')

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0

    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
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

theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of Iterations')
pyplot.ylabel('Cost J')
pyplot.title('Food Truck Profits and Population Density')
plt.show()

print('theta computed from gradient descent: {:s}'.format(str(theta)))

X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price),'\n')

# Normal Equation
pathdata = 'Square Feet, Bedrooms, Housing Prices.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')

X = data[:, :2]
y = data[:, 2]
m = y.size

X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    # theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    theta = np.matmul(np.transpose(X),X)
    theta = np.linalg.inv(theta)
    theta = np.matmul(theta,X.T)
    theta = np.matmul(theta,y)
    return theta

theta = normalEqn(X, y);

# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta)));

# Estimate the price of a 1650 sq-ft, 3 br house
X_array = [1, 1650, 3]
price = np.dot(X_array, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price),'\n')