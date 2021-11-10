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
print('\nTheta computed from the normal equations: {:s}'.format(str(theta)));

# Estimate the price of a 1650 sq-ft, 3 br house
X_array = [1, 1650, 3]
price = np.dot(X_array, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price),'\n')