# Shift + Alt + A

import sys
import os

import matplotlib.pyplot as plt

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Scientific and vector computation for python
import numpy as np

from scipy import optimize

scriptpath = "C:/Users/mackt/Python/Machine Learning/3. Microchips/utils"
sys.path.append(os.path.abspath(scriptpath))

# Load data
# The first two columns contains the exam scores and the third column contains the label.
pathdata = 'Microchip Tests, Pass QA.txt'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = np.loadtxt(path, delimiter=',')
X, y = data[:, 0:2], data[:, 2:3]

def sigmoid(X):
    return 1 /(1 + np.exp(-X))

def mapFeature(X1, X2):
    degree = 6
    if type(X1) is np.float64:
        out = np.ones([1,1])
    else:
        out = np.ones([X1.shape[0],1])
    for i in range(1,degree+1):
        for j in range(0,i+1):
            new_column = (X1 ** (i-j)) * (X2 ** j)
            out = np.column_stack([out,new_column])
    return out

# J(theta)
def costFunction(theta, X, y, lambda_):
    theta = np.reshape(theta,(X.shape[1],1))
    sig = sigmoid(np.dot(X,theta))
    m = X.shape[0]
    cost = (np.dot(-y.T,np.log(sig)) - np.dot(1-y.T,np.log(1-sig))) / m
    cost = cost + np.dot(theta.T[0,1:],theta[1:,0]) * lambda_ / (2 * m)
    return cost

# d J(theta)/d theta
def gradient(theta, X, y, lambda_):
    theta = np.reshape(theta,(X.shape[1],1))
    m = X.shape[0]
    sig = sigmoid(np.dot(X,theta))
    theta[0] = 0
    grad = np.zeros([X.shape[1],1])
    grad = np.dot(X.T,(sig - y)) / m
    grad = grad + theta * lambda_ / m
    return grad

def plotDecisionBoundary(theta, X, y):
    (m,n) = X.shape
    if n <= 3:
        X1_min = np.min(X[:,1])
        X1_max = np.max(X[:,1])
        X1 = np.arange(X1_min, X1_max,0.5)
        X2 = -(theta[0] + theta[1] * X1) / theta[2]
        plt.plot(X1,X2,'-')
    else:
        X1 = np.linspace(-1, 1.5, 50)
        X2 = np.linspace(-1, 1.5, 50)
        Z = np.zeros([X1.shape[0],X2.shape[0]])
        for i in range(0,len(X1)):
            for j in range(0,len(X2)):
                Z[i,j] = mapFeature(X1[i],X2[j]).dot(theta)
        Z = Z.T
        plt.contourf(X1,X2,Z,0, colors = ('r','g'), alpha = 0.2)

def plotdata(X, y):
    postive = np.where(y > 0.5)
    negtive = np.where(y < 0.5)
    plt.scatter(X[postive[0],0],X[postive[0],1],marker='+',c='g')
    plt.scatter(X[negtive[0],0],X[negtive[0],1],marker='x',c='r')
    pyplot.title('QA Testing Results')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    pyplot.legend(['Pass', 'Fail'], loc='upper right')

def traccuracy(X,y,theta):
    H = sigmoid(np.dot(Xmapped,theta))
    index = np.where(H>=0.5)
    p = np.zeros([m,1])
    p[index] = 1
    prob = np.mean(p == y) * 100
    return prob

# Plot Data
plotdata(X, y)
plt.show()

# Size of X and y
print('\nX:',X.shape)
print('y:',y.shape)

# Increase number of features
# n=2 -> n =28
# Xmapped = [1 x1 x2 x1^2 x1x2 x2^2 ... x1x2^5 x2^6]
Xmapped = mapFeature(X[:,0], X[:,1])
(m,n) = Xmapped.shape
print('\nm:',m,'\nn:',n,'\n')

# Initial Theta
lambda_ = 1
initial_theta = np.zeros([n,1])
cost = costFunction(initial_theta, Xmapped, y, lambda_)
grad = gradient(initial_theta, Xmapped, y, lambda_)
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})
print('Cost at initial theta (zeros): {:.3f}'.format(cost[0][0]))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only: '
        ,*np.reshape(grad[0:5], (1,5)))
print('Expected gradients (approx) - first five values only:'
        ' [0.0085 0.0188 0.0001 0.0503 0.0115]\n')

# Test Theta
lambda_ = 10
test_theta = np.ones([n,1])
cost = costFunction(test_theta, Xmapped, y, lambda_)
grad = gradient(test_theta, Xmapped, y, lambda_)
print('Cost at test theta (with lambda = 10): {:.2f}'.format(cost[0][0]))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only: '
        ,*np.reshape(grad[0:5], (1,5)))
print('Expected gradients (approx) - first five values only:'
        ' [0.3460 0.1614 0.1948 0.2269 0.0922]\n')

# Optimize Theta for different lambdas
lambda_ = [0.2,1,10,100]
fig = pyplot.figure(figsize=(15, 15))
i = 0
for i in range (0,4):
    initial_theta = np.zeros([n,1])
    result =  optimize.minimize(fun=costFunction,
                                x0=initial_theta,
                                args=(Xmapped,y,lambda_[i]),
                                method='TNC',
                                jac=gradient)
    theta = result.x
    cost = result.fun
    if i == 0:
        ax = pyplot.subplot(221)
        plotdata(X,y)
        plotDecisionBoundary(theta, Xmapped, y)
        pyplot.title('Lambda = {:.2f}'.format(lambda_[i]))
    elif i == 1:
        ax = pyplot.subplot(222)
        plotdata(X,y)
        plotDecisionBoundary(theta, Xmapped, y)
        pyplot.title('Lambda = {:.2f}'.format(lambda_[i]))

        prob = traccuracy(Xmapped,y,theta)

        print('Training Set Accuracy: {:.1f}%'.format(prob))
        print('Expected Accuracy (lambda = 1): 83.1 %\n')
    elif i == 2:
        ax = pyplot.subplot(223)
        plotdata(X,y)
        plotDecisionBoundary(theta, Xmapped, y)
        pyplot.title('Lambda = {:.2f}'.format(lambda_[i]))
    elif i == 3:
        ax = pyplot.subplot(224)
        plotdata(X,y)
        plotDecisionBoundary(theta, Xmapped, y)
        pyplot.title('Lambda = {:.2f}'.format(lambda_[i]))
plt.show()