import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat

def linearRegCostFunction(Theta, X, y, Lambda):
    m = X.shape[0]

    J = 1.0 / (2 * m) * np.sum(np.square(X.dot(Theta) - y)) + 1.0 * Lambda / (2 * m) * np.sum(np.square(Theta[1:]))

    mask = np.eye(len(Theta))
    mask[0, 0] = 0
    grad = 1.0 / m * X.T.dot(X.dot(Theta) - y) + 1.0 * Lambda / m * (mask.dot(Theta))

    return J, grad

def trainLinearReg(X, y, Lambda):
    initial_theta = np.zeros(X.shape[1])
    initial_theta = np.reshape(initial_theta, (-1,1))
    res = optimize.minimize(fun = linearRegCostFunction,
                            x0 = initial_theta,
                            args = (X, y, Lambda),
                            jac=True,
                            method='TNC',
                            options={'maxiter': 200})

    return res.x

def learningCurve(X, y, Xval, yval, Lambda):
    m = X.shape[0]
    m_val = Xval.shape[0]

    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    for i in range(1, m + 1):
        theta = trainLinearReg(X[:i, ], y[:i, ], Lambda)
        error_train[i - 1] = 1.0 / (2 * i) * np.sum(np.square(X[:i, ].dot(theta) - y[:i, ]))
        error_val[i - 1] = 1.0 / (2 * m_val) * np.sum(np.square(Xval.dot(theta) - yval))

    return error_train, error_val

def polyFeatures(X, p):
    # X_poly[i, :] = [X[i], X[i]**2, X[i]**3 ...  X[i]**p]
    X_poly = np.zeros((len(X), p))

    for i in range(p):
        X_poly[:, i] = np.power(X, i + 1).ravel()

    return X_poly

def featureNormalize(X, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, ddof=1, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, Theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly, dummy_mu, dummy_sigma = featureNormalize(X_poly, mu, sigma)
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, Theta), '--', lw=2)

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))
    m = X.shape[0]
    mval = Xval.shape[0]

    for i in range(len(lambda_vec)):
        Lambda = lambda_vec[i]
        Theta = trainLinearReg(X, y, Lambda)
        error_train[i] = 1.0 / (2 * m) * np.sum(np.square(X.dot(Theta) - y))
        error_val[i] = 1.0 / (2 * mval) * np.sum(np.square(Xval.dot(Theta) - yval))
    return lambda_vec, error_train, error_val

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Loading and Visualizing Data ========================
    print('='*10, 'Part 1: Loading and Visualizing Data', '='*10)
    print("\nLoading Data...\n")

    pathdata = 'Dam Water Flow.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    # Training Data (60%)
    X = data['X']
    y = data['y'].ravel()
    m = X.shape[0]
    print("There are {} Training Examples.".format(m))

    # Test Data (20%)
    Xtest = data['Xtest']
    ytest = data['ytest'].ravel()
    mtest = Xtest.shape[0]
    print("There are {} Test Examples.".format(mtest))

    # Validation Data (20%)
    Xval = data['Xval']
    yval = data['yval'].ravel()
    mval = Xval.shape[0]
    print("There are {} Validation Examples.\n".format(mval))

    # Plot training data
    plt.plot(X, y, 'ro', ms=10, mec='k', mew=1)
    plt.xlabel('Change in water level (mm)')
    plt.ylabel('Water flowing out of the dam (KL/s)')
    plt.title('Dam Water Flow and Water Level')
    plt.show()

    # ========================= Part 2: Linear Regression Cost ========================
    print('='*10, 'Part 2: Linear Regression Cost', '='*10)

    Lambda = 1
    Theta = np.array([1, 1])
    J, _ = linearRegCostFunction(Theta, np.hstack((np.ones((m, 1)), X)), y, Lambda)

    print('\nCost at Theta = [1, 1]: %f' % J)
    print('Cost at Theta = [1, 1]: 303.993192 (Expected)\n' % J)

    # ========================= Part 3: Linear Regression Gradient ========================
    print('='*10, 'Part 3: Linear Regression Gradient', '='*10)

    J, grad = linearRegCostFunction(Theta, np.hstack((np.ones((m, 1)), X)), y, Lambda)
    print('\nGradient at Theta = [1, 1]: [{:.6f}, {:.6f}]'.format(grad[0], grad[1]))
    print('Gradient at Theta = [1, 1]: [-15.303016, 598.250744] (Expected)\n')

    # ========================= Part 4: Training Linear Regression ========================
    print('='*10, 'Part 4: Training Linear Regression', '='*10)

    Theta = trainLinearReg(np.hstack((np.ones((m, 1)), X)), y, Lambda)
    pred = np.hstack((np.ones((m, 1)), X)).dot(Theta)

    print("\nPlotting Linear Regression...\n")
    # print(Theta)
    plt.plot(X, y, 'ro', ms=10, mec='k', mew=1)
    plt.xlabel('Change in water level (mm)')
    plt.ylabel('Water flowing out of the dam (KL/s)')
    plt.plot(X, pred, 'b', lw=2)
    plt.title('Dam Water Flow and Water Level')
    plt.show()


    # ========================= Part 5: Learning Curve for Linear Regression ========================
    print('='*10, 'Part 5: Learning Curve for Linear Regression', '='*10)

    Lambda = 0
    X_aug = np.hstack((np.ones((m, 1)), X))
    Xval_aug = np.hstack((np.ones((mval, 1)), Xval))
    error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, Lambda)

    plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()

    print('\n# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%.3f\t%.3f' % (i+1, error_train[i], error_val[i]))

    print('\nThe Training Error and Cross Validation are high when the number of training examples')
    print('are increased. Therefore, the model suffers from HIGH BIAS or UNDERfitting.\n')
    print('The issue can be solved using Polynomial Regression and adding more features.\n')

    # =========== Part 6: Feature Mapping for Polynomial Regression =============
    print('='*10, 'Part 6: Feature Mapping for Polynomial Regression', '='*10)

    # X_poly[i, :] = [X[i], X[i]**2, X[i]**3 ...  X[i]**p]
    p = 8 # Polynomial Degree

    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test -= mu
    X_poly_test /= sigma
    X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val -= mu
    X_poly_val /= sigma
    X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

    # There are m training examples
    np.set_printoptions(precision = 2)
    print('\nNormalized Training Example 1: ',X_poly[0, :])
    print('Normalized Training Example 2: ',X_poly[1, :], '\n')

    # ========================= Part 7: Polynomial Regression ========================
    print('='*10, 'Part 7: Polynomial Regression', '='*10, "\n")

    # ------------------- Training ---------------------
    # Poly_Reg = {}
    for Lambda in [0,1,100,1100]:
        # Train Polynomial Regression
        print('Learning Polynomial Regression...\n')
        Theta = trainLinearReg(X_poly, y, Lambda)
        # Poly_Reg["Lambda{0}".format(Lambda)] = Theta

        # Plot training data and fit
        fig = plt.figure(figsize=(12, 5))

        ax = fig.add_subplot(121)
        print('Plotting Polynomial Regression...')
        plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
        plotFit(np.min(X), np.max(X), mu, sigma, Theta, p)
        plt.xlabel('Change in water level (mm)')
        plt.ylabel('Water flowing out of the dam (KL/s)')
        plt.title('Polynomial Regression Fit (lambda = %i)' % Lambda)
        plt.ylim([-20, 50])

        # ------------------- Learning Curve -----------------------
        error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)

        # Plot Learning Curve
        ax = fig.add_subplot(122)
        plt.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)
        plt.title('\nPolynomial Regression Learning Curve (Lambda = %i)' % Lambda)
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.axis([0, 13, -5, 160])
        plt.legend(['Train', 'Cross Validation'])
        plt.show()

        print('\n------ Polynomial Regression (Lambda = %i) ------\n' % Lambda)
        print('# Training Examples\tTrain Error\tCross Validation Error')
        for i in range(m):
            print('  \t%d\t\t%.3f\t\t%.3f' % (i+1, error_train[i], error_val[i]))
        print('\n')

# =========== Part 8: Selecting Lambda =============
    print('='*10, 'Part 8: Selecting Lambda', '='*10)

    Delta = []
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    plt.figure()
    plt.plot(lambda_vec, error_train, color='b', label='Train')
    plt.plot(lambda_vec, error_val, color='r', label='Cross Validation')
    plt.legend(loc='upper right')
    plt.title('\nSelecting Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.show()

    print('\nLambda\tTrain Error\tCross Validation Error\tDelta')
    for i in range(len(lambda_vec)):
        Delta.append(abs(error_train[i] - error_val[i]))
        print('%.3f\t%.3f\t\t%.3f\t\t\t%.3f' % (lambda_vec[i], error_train[i], error_val[i], abs(error_train[i] - error_val[i])))

    Lambda = lambda_vec[Delta.index(min(Delta))]

    print("\nThe most appropriate Lambda to use is Lambda = %.3f.\n" % Lambda)

    print('Learning Polynomial Regression...\n')
    Theta = trainLinearReg(X_poly, y, Lambda)

    fig = plt.figure(figsize=(12, 5))

    # ------------------- Training ---------------------
    ax = fig.add_subplot(121)
    print('Plotting Polynomial Regression...')
    plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
    plotFit(np.min(X), np.max(X), mu, sigma, Theta, p)
    plt.xlabel('Change in water level (mm)')
    plt.ylabel('Water flowing out of the dam (KL/s)')
    plt.title('Polynomial Regression Fit (lambda = %.3f)' % Lambda)
    plt.ylim([-20, 50])

    # ------------------- Learning Curve -----------------------
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)

    # Plot Learning Curve
    ax = fig.add_subplot(122)
    plt.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)
    plt.title('\nPolynomial Regression Learning Curve (Lambda = %.3f)' % Lambda)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, -5, 160])
    plt.legend(['Train', 'Cross Validation'])
    plt.show()

    print('\n------ Polynomial Regression (Lambda = %i) ------\n' % Lambda)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%.2f\t\t%.2f' % (i+1, error_train[i], error_val[i]))
    print('\n')

# =========== Part 9: Computing Test Error =============
    print('='*10, 'Part 9: Computing Test Error', '='*10)

    Lambda = 3.00

    print('\nLearning Polynomial Regression...\n')
    Theta = trainLinearReg(X_poly, y, Lambda)

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    print('Plotting Polynomial Regression...')
    plt.plot(Xtest, ytest, 'ro', ms=10, mew=1.5, mec='k')
    plotFit(np.min(X), np.max(X), mu, sigma, Theta, p)
    plt.xlabel('Change in water level (mm)')
    plt.ylabel('Water flowing out of the dam (KL/s)')
    plt.title('Polynomial Regression Fit (lambda = %.3f)' % Lambda)
    plt.ylim([-20, 50])

    # ------------------- Learning Curve -----------------------
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
    error_train, error_test = learningCurve(X_poly, y, X_poly_test, ytest, Lambda)

    # Plot Learning Curve
    ax = fig.add_subplot(122)
    plt.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val, np.arange(1, 1+m), error_test)
    plt.title('\nPolynomial Regression Learning Curve (Lambda = %.3f)' % Lambda)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, -5, 160])
    plt.legend(['Train', 'Cross Validation', 'Test'])
    plt.show()

    print('\n------ Polynomial Regression (Lambda = %i) ------\n' % Lambda)
    print('# Training Examples\tTrain Error\tCross Validation Error\tTest Error')
    for i in range(m):
        print('  \t%d\t\t%.2f\t\t%.2f\t\t\t%.2f' % (i+1, error_train[i], error_val[i],error_test[i]))
    print('\n')

    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
    lambda_vec, error_train, error_test = validationCurve(X_poly, y, X_poly_test, ytest)
    i = np.where(lambda_vec==Lambda)
    print('Lambda\tTrain Error\tCross Validation Error\tTest Error')
    print('%.3f\t%.3f\t\t%.3f\t\t\t%.3f' % (lambda_vec[i], error_train[i], error_val[i], error_test[i]))

    print('\n','='*22, "End", '='*22)