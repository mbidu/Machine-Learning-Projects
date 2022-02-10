import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Loading and Visualizing Data ========================
    print('='*10, 'Part 1: Loading and Visualizing Data', '='*10)
    print("\nLoading Data...\n")

    pathdata = 'Dam Water Flow.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)