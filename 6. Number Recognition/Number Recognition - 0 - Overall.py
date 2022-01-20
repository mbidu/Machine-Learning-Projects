pathdata = 'ex3weights.mat'
path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
data = loadmat(path)
Theta1 = data['Theta1'] # (25, 401)
Theta2 = data['Theta2'] # (10, 26)