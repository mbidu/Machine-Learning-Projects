import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from scipy.io import loadmat
import math

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

def sigmoid(z):
    return 1 /(1 + np.exp(-z))

def predict(Theta1, Theta2, X):
    m, n = X.shape
    num_labels = Theta2.shape[0]
    p = np.zeros((m,1))

    # Add ones to X matrix
    X = np.hstack((np.ones((m,1)),X))
    a2 = sigmoid(X.dot(Theta1.T))
    a2 = np.hstack((np.ones((m,1)), a2))
    a3 = sigmoid(a2.dot(Theta2.T))
    p = np.argmax(a3, axis=1)
    return p

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

    print('='*40)
    print('\nLoading Neural Network...\n')

    pathdata = 'Neural_Network_Parameters.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    para = loadmat(path)
    Theta1 = para['Theta1'] # (25, 401)
    Theta2 = para["Theta2"] # (10, 26)

    print('='*40)

    pred = predict(Theta1, Theta2, X) + 1
    acc = np.mean(pred == y.flatten())
    print("\nTraining Set Accuracy: ", acc, "\n")

    print('='*40)

    rp = np. random.choice(m, size = 10, replace = False)
    for i in range(3):
        print("\nDisplaying Image...")
        x = X[rp[i], :].reshape(1,n) # (1,400)
        pred = predict(Theta1, Theta2, x) + 1
        print("Neural Network Prediction: ", np.mod(pred,10))
        if y[rp[i]] != 10:
            print("The Class is: ", y[rp[i]])
        else:
            print("The Class is: 0")
        fig = plt.figure()
        x = x.reshape(20,20)
        x = ndimage.rotate(x, -90)
        x = np.fliplr(x)
        plt.imshow(x, cmap='binary')
        plt.show()

    print('\n','='*22, "End", '='*22)
