# Shift + Alt + A

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.io import loadmat
import math
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
            ax[row, column].matshow(X[rows*row+column].reshape(20,20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])

input_layer_size  = 400 # 20x20 Input Images of Digits
hidden_layer_size = 25 #25 hidden units
num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

print('='*18, 'Start', '='*18)
print('\nDisplaying Data...\n')

pathdata = 'Handwritten_Characters.mat'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = loadmat(path)
X, y = data['X'], data['y']
m, n = X.shape
y = np.reshape(y, (y.shape[0],1))

# All Data
plt.title('All Handwritting Character Examples')
plt.imshow(X, cmap = 'binary') # Binary = black & white

# Random 100 Characters
index = np.random.choice(m,size=100, replace=False)
print(index)
print(y[index, 0])
displaydata(X[index])

# 10 Character Classes
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
    ax[-1].set_title("Class:"+str(i))  # set title
    plt.imshow(x.reshape(20,20), cmap='binary')
print(ClassEx, '\n')
plt.show()

print('='*18, 'Finish', '='*18)