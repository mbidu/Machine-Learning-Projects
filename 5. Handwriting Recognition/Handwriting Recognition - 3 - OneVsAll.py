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

def OneVsAll(X, y, num_labels, Mylambda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n+1))
    X = np.column_stack((np.ones((X.shape[0],1)), X))
    for i in range (1, num_labels+1):
        print('Learning Class:', i)
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(y_i.shape[0], 1)
        ret = optimize.minimize(fun = LogisticRegression_CostFunction_J,
                            x0 = theta,
                            args = (X, y_i, Mylambda),
                            method = 'TNC',
                            jac = LogisticRegression_CostFunction_grad,
                            options = {'disp': False})
        all_theta[i-1, :] = ret.x
    return all_theta

def Predict_OneVsAll(all_theta, X):
    m, n = X.shape
    num_labels = all_theta.shape[0]
    p = np.zeros((m,1))
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    h = sigmoid(X.dot(all_theta.T))
    prediction = np.argmax(h, axis = 1) + 1
    return prediction

if __name__ == '__main__':

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    print('='*18, "Beginning", '='*18)

    pathdata = 'Handwritten_Characters.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)
    X, y = data['X'], data['y']
    m, n = X.shape
    y = np.reshape(y, (y.shape[0],1))

    # Training One Vs All Models
    print('\n', '='*5, 'Training One Vs All Model', '='*5, '\n')

    Mylambda = 5
    all_theta = OneVsAll(X, y, num_labels, Mylambda)
    # print('\nMylambda =', Mylambda)
    # print('Trained parameters (all_theta) matrix shape is', num_labels,'x', n+1)

    # Class Accuracy
    print('\n', '='*12, 'Class Accuracy', '='*12)

    prediction = Predict_OneVsAll(all_theta, X)
    prediction = np.reshape(prediction, (prediction.shape[0],1))

    for i in range(1, num_labels + 1):
        pred_i = 0
        for j in range (prediction.shape[0]):
            if prediction[j,0] == y[j,0] and prediction[j,0] == i:
                pred_i = pred_i + 1
        y_i = np.sum(y == i)
        acc_i = 100*(pred_i / y_i)

        print('\nLearning Class: ', i)
        print('Accuracy = {:.2f}%'.format(acc_i))
        print('Actual Cases = ', y_i)

    # Overall Accuracy
    print('\n', '='*12, 'Overall Results', '='*12)

    accuracy = np.mean(prediction == y)
    Correct_pred = np.sum(prediction == y)
    print('\nCorrect Training Set Predictions: {}'.format(Correct_pred))
    print('Training Set Accuracy: {:.2f}%\n'.format(accuracy*100))

    print('='*22, "End", '='*22)