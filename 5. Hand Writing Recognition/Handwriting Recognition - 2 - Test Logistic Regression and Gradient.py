# Shift + Alt + A

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.io import loadmat
import math
import pandas as pd

def sigmoid(X):
    return 1 /(1 + np.exp(-X))

def LogisticRegression_CostFunction_J(theta_t, X_t, y_t, lambda_t):
    m = y_t.shape[0]
    theta = theta_t.copy()
    theta[0] = 0

    h = sigmoid(X_t.dot(theta_t))
    h = np.reshape(h, (h.shape[0],1))
    J = lambda_t/(2*m)*np.sum(np.square(theta)) - 1/m * np.sum((y_t*np.log(h)) + (1-y_t)*(np.log(1-h)))
    return J

def LogisticRegression_CostFunction_grad(theta_t, X_t, y_t, lambda_t):
    theta_t = theta_t.reshape(theta_t.shape[0], 1)
    m = y_t.shape[0]
    theta = theta_t.copy()
    theta[0] = 0

    h = sigmoid(X_t.dot(theta_t))
    h = np.reshape(h, (h.shape[0],1))
    # grad = X_t.T.dot(h - y_t)/m + lambda_t / m * theta_t
    grad = (X_t.T).dot(h - y_t)/m + lambda_t / m * theta
    return grad.flatten()

print('='*18, 'Start', '='*18)

pathdata = 'Handwritten_Characters.mat'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = loadmat(path)
X, y = data['X'], data['y']
m, n = X.shape
y = np.reshape(y, (y.shape[0],1))

# Test values
theta_t = np.array([-2,-1,1,2])
theta_t = theta_t.reshape((theta_t.shape[0],1))
X_t = np.column_stack((np.ones((5, 1)), (np.array([range(1, 16)])/10).reshape(3, 5).T))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5,1)
lambda_t = 3
print('\nTest Theta\n', theta_t)
print('Test X\n', X_t)
print('Test y\n', y_t)
print('Test Lambda\n', lambda_t)

J = LogisticRegression_CostFunction_J(theta_t, X_t, y_t, lambda_t)
grad = LogisticRegression_CostFunction_grad(theta_t, X_t, y_t, lambda_t)
print('\nCost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('Gradients         : [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients: [0.146561, -0.548558, 0.724722, 1.398003]\n');

print('='*18, 'Finish', '='*18)