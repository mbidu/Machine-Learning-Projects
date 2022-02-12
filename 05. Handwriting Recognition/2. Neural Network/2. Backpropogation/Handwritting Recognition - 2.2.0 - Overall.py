import os
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy.io import loadmat
from scipy import optimize
import math

from scipy import ndimage
import pandas as pd

######################### DEFINITIONS ##############################
# ================== 1. Data ===================
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

# ================= Functions =====================
def sigmoid(z):
    return 1 /(1 + np.exp(-z))

def sigmoidgradient(z):
    return sigmoid (z) * (1 - sigmoid(z))

# ================= Cost Function =====================
def nnCostFunction (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Labmda):
    # --------------- Common Parameters ----------------
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 = Theta1.reshape(hidden_layer_size, input_layer_size + 1)  # (25, 401)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape(num_labels, hidden_layer_size + 1) # (10, 26)

    temp1 = Theta1.copy() # (25, 401)
    temp2 = Theta2.copy() # (10, 26)
    temp1[:, 0] = 0
    temp2[:, 0] = 0

    m, n = X.shape
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)

    Y = np.zeros((X.shape[0], num_labels)) # (5000, 10)
    Y[np.arange(m), y.flatten() - 1] = 1

    # ---------------- J -----------------
    a1 = np.hstack((np.ones((m, 1)), X)) # (5000, 401)
    z2 = a1.dot(Theta1.T) # (5000, 25)
    a2 = sigmoid(z2) # (5000, 25)
    a2 = np.hstack((np.ones((m, 1)), a2)) # (5000, 26)
    z3 = a2.dot(Theta2.T) # (5000, 10)
    a3 = sigmoid(z3) # (5000, 10)
    h = a3 # (5000, 10)

    reg_term = Lambda / (2*m) * (np.sum(temp1**2) + np.sum(temp2**2))
    J = (1/m) * np.sum(np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h), axis = 1))
    J += reg_term

    # -------------------- grad --------------------
    D1 = np.zeros_like(Theta1)
    D2 = np.zeros_like(Theta2)

    for i in range(m):
        a1 = X[i, :]
        a1 = np.hstack((1, a1))
        a2 = sigmoid(Theta1.dot(a1))
        a2 = np.hstack((1, a2))
        a3 = sigmoid(Theta2.dot(a2))

        delta3 = a3 - Y[i, :]
        delta2 = Theta2.T.dot(delta3)
        delta2 = delta2[1:] * sigmoidgradient(Theta1.dot(a1))

        D2 += np.dot(delta3.reshape(delta3.shape[0], 1), a2.reshape(1, a2.shape[0])) # (10, 26)
        D1 += np.dot(delta2.reshape(delta2.shape[0], 1), a1.reshape(1, a1.shape[0])) # (25, 401)

    grad_item1 = Lambda / m
    grad_item1 = grad_item1 * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad_item2 = Lambda / m
    grad_item2 = grad_item2 * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    Theta1_grad = (1/m) * D1
    Theta1_grad += grad_item1
    Theta2_grad = (1/m) * D2
    Theta2_grad += grad_item2

    grad = np.vstack((Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1)))

    return J, grad

# ================= 6. Random Initial Weights =====================
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_in, L_out + 1))
    epsilon_init = 0.12
    W = np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init
    return W

# ================= 7. Checking Gradients =====================
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(np.arange(1, W.size + 1)), W.shape) / 10
    return W

def check_NN_Gradients(Lambda):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m+1), num_labels)

    # Unroll Parameters
    nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

# def computeNumericalGradient(J, Theta)
    numgrad = np.zeros((nn_params.shape))
    perturb = np.zeros(nn_params.shape)
    e = 1e-4

    for p in range (nn_params.size):
        perturb[p] = e
        loss1, grad1 = nnCostFunction(nn_params-perturb, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        loss2, grad2 = nnCostFunction(nn_params+perturb, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    # for i in range(numgrad.size):
    #     print('{}-----{}'.format(numgrad[i], grad[i]))

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    # print('\nThe above two columns should be very similar.')
    # print('Numerical Gradient (Left) ----- Analytical Gradient (Right)')
    # print('Relative Difference: ', diff)
    if diff <1e-9:
        print('CORRECT Backpropagation Implementation\n')
    else:
        print('INCORRECT Backpropagation Implementation\n')

# ================= 8. Optimizing (Training) Neural Network =====================
def nnCost (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Labmda):
    # --------------- Common Parameters ----------------
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 = Theta1.reshape(hidden_layer_size, input_layer_size + 1)  # (25, 401)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape(num_labels, hidden_layer_size + 1) # (10, 26)

    temp1 = Theta1.copy() # (25, 401)
    temp2 = Theta2.copy() # (10, 26)
    temp1[:, 0] = 0
    temp2[:, 0] = 0

    m, n = X.shape
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)

    Y = np.zeros((X.shape[0], num_labels)) # (5000, 10)
    Y[np.arange(m), y.flatten() - 1] = 1

    # ---------------- J -----------------
    a1 = np.hstack((np.ones((m, 1)), X)) # (5000, 401)
    z2 = a1.dot(Theta1.T) # (5000, 25)
    a2 = sigmoid(z2) # (5000, 25)
    a2 = np.hstack((np.ones((m, 1)), a2)) # (5000, 26)
    z3 = a2.dot(Theta2.T) # (5000, 10)
    a3 = sigmoid(z3) # (5000, 10)
    h = a3 # (5000, 10)

    reg_term = Lambda / (2*m) * (np.sum(temp1**2) + np.sum(temp2**2))
    J = (1/m) * np.sum(np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h), axis = 1))
    J += reg_term

    return J

def nnGrad (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Labmda):
    # --------------- Common Parameters ----------------
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 = Theta1.reshape(hidden_layer_size, input_layer_size + 1)  # (25, 401)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape(num_labels, hidden_layer_size + 1) # (10, 26)

    temp1 = Theta1.copy() # (25, 401)
    temp2 = Theta2.copy() # (10, 26)
    temp1[:, 0] = 0
    temp2[:, 0] = 0

    m, n = X.shape
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)

    Y = np.zeros((X.shape[0], num_labels)) # (5000, 10)
    Y[np.arange(m), y.flatten() - 1] = 1

    # -------------------- grad --------------------
    D1 = np.zeros_like(Theta1)
    D2 = np.zeros_like(Theta2)

    for i in range(m):
        a1 = X[i, :]
        a1 = np.hstack((1, a1))
        a2 = sigmoid(Theta1.dot(a1))
        a2 = np.hstack((1, a2))
        a3 = sigmoid(Theta2.dot(a2))

        delta3 = a3 - Y[i, :]
        delta2 = Theta2.T.dot(delta3)
        delta2 = delta2[1:] * sigmoidgradient(Theta1.dot(a1))

        D2 += np.dot(delta3.reshape(delta3.shape[0], 1), a2.reshape(1, a2.shape[0])) # (10, 26)
        D1 += np.dot(delta2.reshape(delta2.shape[0], 1), a1.reshape(1, a1.shape[0])) # (25, 401)

    grad_item1 = Lambda / m
    grad_item1 = grad_item1 * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad_item2 = Lambda / m
    grad_item2 = grad_item2 * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    Theta1_grad = (1/m) * D1
    Theta1_grad += grad_item1
    Theta2_grad = (1/m) * D2
    Theta2_grad += grad_item2

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad

def predict(Theta1, Theta2, X):
    m, n = X.shape
    num_labels = Theta2.shape[0]
    p =np.zeros((m,1))

    h1 = sigmoid(np.hstack((np.ones((m,1)), X)).dot(Theta1.T)) # (5000, 25)
    h2 = sigmoid(np.hstack((np.ones((m,1)), h1)).dot(Theta2.T)) # (5000, 10)
    p = np.argmax(h2, axis = 1)

    return p

######################### MAIN ##############################

if __name__ == '__main__':

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # ========= 1. Loading and Visualizing Data ============
    # Load Training Data
    print('='*18, "Beginning", '='*18)
    print("\nLoading Data...\n")

    pathdata = 'Handwritten_Characters2.mat'
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
    displaydata(X[index])
    plt.show()

    # ========= 2. Loading Pre-Determined Parameters ============
    # Load Pre-Determined Parameters
    print('='*40)
    print("\nLoading Pre-Determined Parameters...\n")

    pathdata = 'Neural_Network_Parameters2.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    pre_weights = loadmat(path)

    Theta1 = pre_weights['Theta1'] # (25, 401)
    Theta2 = pre_weights['Theta2'] # (10, 26)

    nn_params = np.vstack((np.reshape(Theta1, (-1, 1)), np.reshape(Theta2, (-1, 1))))

    # ========= 3. Feedforward - Compute Cost ============
    print('='*40)

    # Weight Regularization Parameter
    Lambda = 0
    # Cost Function
    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    print('\nCost at Pre-Determined Parameters: %.6f' % J)
    print('Cost at Pre-Determined Parameters: 0.287629 (Expected)\n')

    # ========= 4. Feedforward - Compute Cost with Regularization ============
    print('='*40)

    # Weight Regularization Parameter
    Lambda = 1
    # Cost Function
    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    print('\nCost at Pre-Determined Parameters: %.6f' % J)
    print('Cost at Pre-Determined Parameters: 0.383770 (Expected)\n')

    # ========= 5. Testing Sigmoid Gradient ============
    print('='*40)
    print("\nEvaluating Sigmoid Gradient...\n")

    test_num = np.array([-1, -0.5, 0, 0.5, 1])
    g = sigmoidgradient(test_num)
    round_g = [round(num, 3) for num in g]

    print('Sigmoid Gradient at [-1 -0.5 0 0.5 1]: ', round_g, '\n')

    # ========= 6. Backpropagation - Initializing Pre-Set Parameters ============
    print('='*40)
    print("\nInitializing Neural Network Pre-Set Parameters...\n")

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # unroll parameters
    initial_nn_params = np.vstack((np.reshape(initial_Theta1, (-1, 1)), np.reshape(initial_Theta2, (-1, 1))))

    # ========= 7. Backpropagation - Backpropagation ============
    print('='*40)
    print("\nChecking Backpropagation...\n")

    check_NN_Gradients(Lambda)

    # ========= 8. Backpropagation - Regularization ============
    print('='*40)
    print("\nChecking Backpropagation with Regularization...\n")

    Lambda = 3
    check_NN_Gradients(Lambda)

    debug_J, debug_grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    print("Cost at fixed debugging parameters with Lambda = {}: %.6f".format(Lambda) % debug_J)
    print("Cost at fixed debugging parameters with Lambda = 3: 0.576051 (Expected)\n")

    # ========= 9. Backpropagation - Training Neural Network ============
    print('='*40)
    print("\nTraining Neural Network...")
    print("Will take close to 5 minutes...\n")

    Lambda = 1

    # Minimizing using conjugate gradient algorithm
    nnParam = scipy.optimize.fmin_cg(f = nnCost, x0 = initial_nn_params, fprime = nnGrad,
                                        args = (input_layer_size, hidden_layer_size, num_labels, X, y, Lambda),
                                        maxiter = 100, disp = True)

    np.save('Training_Result', nnParam)

    Theta1_cg = np.reshape(nnParam[:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2_cg = np.reshape(nnParam[hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

    # Minimizing using gradient algorithm
    nnParam = optimize.minimize(fun = nnCostFunction,
                                x0 = initial_nn_params,
                                args = (input_layer_size, hidden_layer_size, num_labels, X, y, Lambda),
                                jac=True,
                                method='TNC',
                                options = {'maxiter': 100})

    nnParam = nnParam.x

    Theta1 = np.reshape(nnParam[:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nnParam[hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

    # ========= 10. Backpropagation - Visualizing Weights ============
    print('='*40)
    print("\nVisualizing Neural Network...\n")

    print("Hidden Layer using Conjugate Gradient...")
    displaydata(Theta1_cg[:, 1:])
    plt.show()

    print("Hidden Layer using Gradient...\n")
    displaydata(Theta1[:, 1:])
    plt.show()

    # ========= 11. Backpropagation - Prediction ============
    print('='*40)

    pred = predict(Theta1_cg, Theta2_cg, X) + 1
    acc = np.mean(pred == y.flatten())
    print("\nThe Accuracy using conjugate gradient is about: ", acc)

    pred = predict(Theta1, Theta2, X) + 1
    acc = np.mean(pred == y.flatten())
    print("\nThe Accuracy using gradient is about: ", acc)
    print("The Accuracy using gradient is about:  0.953 +/- 0.01(Expected)\n")

    print('='*22, "End", '='*22)