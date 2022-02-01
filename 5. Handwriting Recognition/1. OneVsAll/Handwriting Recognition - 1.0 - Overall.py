import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from scipy.io import loadmat
import math
from scipy import optimize
import pandas as pd

def displaydata(X, *ex_width):
    if ex_width == ():
        ex_width = round(np.sqrt(X.shape[1]))

    m, n = X.shape
    rows = math.floor(np.sqrt(m))
    cols = math.ceil(m/rows)
    fig, ax = plt.subplots(nrows=rows,
                            ncols=cols,
                            sharey=True,
                            sharex=True,
                            figsize=(8,8))
    fig.suptitle('100 Characters', fontsize=16)

    for row in range(rows):
        for column in range(cols):
            x = X[rows*row+column].reshape(20,20)
            x = ndimage.rotate(x, -90)
            x = np.fliplr(x)
            ax[row, column].matshow(x, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])

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
    print("\nLoading Data...\n")

    pathdata = 'Handwritten_Characters.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)
    # 5000 20pix x 20pix characters
    X = data['X'] # (5000, 400)
    m, n = X.shape
    # Category of each character
    y = data['y'] # (5000, 1)

    # # 100 Random Characters
    print("Loading 100 Random Characters...\n")
    index = np.random.choice(m,size=100, replace=False)
    # print(index)
    # print(y[index, 0])
    displaydata(X[index])
    plt.show()

    # 10 Character Classes
    print("Loading 10 Classes...\n")
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle('10 Classes', fontsize=16)
    ax = []
    ClassEx =[]

    for i in range(1, num_labels + 1):
        j = 0
        for j in range (0, 100 + 1):
            if y[[index[j]], 0] == i:
                break
            else:
                j = j+1
        x = X[index[j], :].reshape(1,n)
        ClassEx.append(index[j])
        ax.append(fig.add_subplot(2, 5, i))
        if i != 10:
            ax[-1].set_title("Class:"+str(i))  # set title
        else:
            ax[-1].set_title("Class: 0")  # set title
        x = x.reshape(20,20)
        x = ndimage.rotate(x, -90)
        x = np.fliplr(x)
        plt.imshow(x, cmap='binary')
    # print(ClassEx, '\n')
    plt.show()

    print('='*5, 'Vectorizing Logistic Regression', '='*5)

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

    # Training One Vs All Models
    print('='*5, 'Training One Vs All Model', '='*5, '\n')

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

    # OneVsAll Accuracy
    print('\n', '='*12, 'OneVsAll Results', '='*12)

    accuracy = np.mean(prediction == y)
    Correct_pred = np.sum(prediction == y)
    print('\nCorrect Training Set Predictions: {}'.format(Correct_pred))
    print('Training Set Accuracy: {:.2f}%\n'.format(accuracy*100))

    print('='*22, "End", '='*22)