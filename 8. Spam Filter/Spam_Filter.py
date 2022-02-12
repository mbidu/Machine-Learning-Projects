import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
from sklearn import svm
from collections import Counter

from process_email import process_email
from email_features import email_features
from get_vocablist import get_vocablist

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # ========================= Part 1: Email Processing ========================
    print('='*10, 'Part 1: Email Processing', '='*10)
    print('\nProcessing Email 1...\n')

    with open(r'C:\Users\mackt\Python\Machine Learning\Data\Email1.txt') as f:
        file_contents = f.read().replace('\n', '')

    word_indices, words, email_sentence = process_email(file_contents)

    # Print Stats
    print('Email...')
    print(file_contents)
    print('\nWords...')
    print(words)
    print('\nEmail String converted with given vocab...')
    print(email_sentence)
    print('\nWord Indices...')
    print(word_indices, '\n')

    # ========================= Part 2: Feature Extraction ========================
    print('='*10, 'Part 2: Feature Extraction', '='*10)
    print('\nExtracting features from Email 1...\n')
    features = email_features(word_indices)

    print('Number of words in vocab:', len(features))
    print('Number of words in Email 1:', len(word_indices))
    print('Number of unique words in Email 1:', np.sum(features > 0),'\n') #print(len(Counter(word_indices).keys()))

    # ========================= Part 3: Train Linear SVM ========================
    print('='*10, 'Part 3: Train Linear SVM', '='*10)
    print('\nProcessing Training Data...\n')

    # 4000 Training Examples of spam and non-spam emails
    pathdata = 'spamTrain.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    # Which of the vocab is used
    X = data['X'] # (4000, 1899)
    # Spam or Not Spam
    y = data['y'].ravel() # (4000,)

    print('Training Linear SVM (Spam Classification)...\n')

    C = 0.1
    clf = svm.LinearSVC(C=C)
    clf.fit(X, y)
    p = clf.predict(X)
    acc = np.mean(p == y) * 100

    print('Training Accuracy: %.2f%%\n' %acc)

    # ========================= Part 4: Test Linear SVM ========================
    print('='*10, 'Part 4: Test Linear SVM', '='*10)
    print('\nProcessing Test Data...\n')

    # 1000 Test Examples of spam and non-spam emails
    pathdata = 'spamTest.mat'
    path = os.path.join('/Users/mackt/Python/Machine Learning/Data', pathdata)
    data = loadmat(path)

    # Which of the vocab is used
    X_test = data['Xtest'] # (1000, 1899)
    # Spam or Not Spam
    y_test = data['ytest'].ravel() # (1000,)

    print('Evaluating Trained Linear SVM with Test Data...\n')

    p = clf.predict(X_test)
    acc = np.mean(p == y_test) * 100

    print('Test Accuracy: %.2f%%\n' %acc)

    # ========================= Part 5: Top Predictors of Spam ========================
    print('='*10, 'Part 5: Top Predictors of Spam', '='*10)
    print('\nGenerating top words found in spam...\n')

    coef = clf.coef_.ravel()
    idx = coef.argsort()[::-1]
    vocab_list = get_vocablist()

    print("Word/Char\tWeight")
    print('---------------------')
    for i in range(15):
        print("{0:<15s} {1:.3f}".format(vocab_list[idx[i]], coef[idx[i]]))

    # ========================= Part 6: Classifying Additional Emails ========================
    print('\n', '='*10, 'Part 6: Classifying Additional Emails', '='*10)

    for filename in ['Spam1.txt', 'Email2.txt', 'Spam2.txt']:
        filepath = os.path.join('/Users/mackt/Python/Machine Learning/Data', filename)

        with open(filepath) as f:
            file_contents = f.read().replace('\n', '')

        word_indices, words, email_sentence = process_email(file_contents)
        x = email_features(word_indices)
        p = clf.predict(x.T)
        if p == 1:
            p = 'Spam'
        elif p == 0:
            p = 'Not Spam'
        print()
        print(filename, 'is', p, '\n')
        print(file_contents)

    print('\n','='*22, "End", '='*22)