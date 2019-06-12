#SCIKIT-LEARN CLASSIFIERS AVAILABLE
  #  KNeighborsClassifier(3),
  #  SVC(kernel="linear", C=0.025),
  #  SVC(gamma=2, C=1),
  #  GaussianProcessClassifier(1.0 * RBF(1.0)),
  #  DecisionTreeClassifier(max_depth=5),
  #  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
  #  MLPClassifier(alpha=1, max_iter=1000),
  #  AdaBoostClassifier(),
  #  GaussianNB(),
  #  QuadraticDiscriminantAnalysis()]
  #  LogisticRegression()

# 6 classifiers to check
from sklearn import tree

# CHALLENGE for Siraj Raval Learn Python for Data Science Video 1: 
# Create 3 more classifiers other than given DecisionTreeClassifier
# I tried 4 
# 1 - SVM
# 2 - Linear Model
# 3 - KNeighborsClassifier
# 4 - GaussianNB
# To compare their accuracy will use accuracy score

from sklearn import svm
# in case it doesn't run without .svm
# from sklearn.svm import SVC

from sklearn import linear_model
# in case it doesn't run without .linear_model
# from sklearn.linear_model import Perceptron

from sklearn import neighbors
# in case it doesn't run without .neighbors
# from sklearn.neighbors import KNeighborsClassifier

from sklearn import naive_bayes
# in case it doesn't run without .naive_bayes
# from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

import numpy as np

# Data and their labels
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Classifiers
# All with default values
clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_linM = linear_model.Perceptron()
clf_ngb = neighbors.KNeighborsClassifier()
clf_nBay = naive_bayes.GaussianNB()

#Training the models
clf_tree = clf_tree.fit(X, Y) # Given in video
clf_svm = clf_svm.fit(X, Y)
clf_linM = clf_linM.fit(X, Y)
clf_ngb = clf_ngb.fit(X, Y)
clf_nBay = clf_nBay.fit(X, Y)

# Testing model using same data - Predict and Check Accuracy
# prediction = clf_tree.predict([[190, 70, 43]]) /  Code from video

pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100    # DecisionTreeClassifier

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100      # SVM

pred_linM = clf_linM.predict(X)
acc_LinM = accuracy_score(Y, pred_linM) * 100    # Linear Model

pred_ngb = clf_ngb.predict(X)
acc_ngb = accuracy_score(Y, pred_ngb) * 100      # KNeighborClassifier

pred_nBay = clf_nBay.predict(X)
acc_nBay = accuracy_score(Y, pred_nBay) * 100    # GaussianNB

# Compare the results

print('Accuracy for Decision Tree: {}'.format(acc_tree)) # Original Model
print('Accuracy for SVM: {}'.format(acc_svm))
print('Accuracy for Linear Model Perceptron: {}'.format(acc_LinM))
print('Accuracy for K Nearest Neighbor: {}'.format(acc_ngb))
print('Accuracy for Naive Bayes Gaussian: {}'.format(acc_nBay))

# Print the best one
index = np.argmax([acc_tree, acc_svm, acc_LinM, acc_ngb, acc_nBay])

# including Decision Tree
classifiers = {0: 'Decision Tree', 1: 'SVM', 2: 'Linear Model - Perceptron', 3: 'K Nearest Neighbor', 4: 'Naive Bayes - GaussianNB'}
print('Best gender classifer is: {}'.format(classifiers[index]))


# ISSUE BELOW
# Tried to modify the index so that Decision Tree is removed but still uses full index by Python
# Need it to pick from ONLY the 4 "new" classifiers
# Tried below code, and variations there of, but no luck
# classifiers_noTree = np.delete(classifiers, [acc_tree])  # 'acc_tree', acc_tree, 0, '0'
# classifiers_noTree = np.delete(classifiers, np.where(classifiers == acc_tree))
# print('Best gender classifer, not considering Decision Tree, is: {}'.format(classifiers_noTree[index]))
# results in Decision Tree being selected everytime so to do test manually removed Decisiion Tree

# The manual way: remove Decision Tree by hand
# Print the best one without including Decision Tree
index = np.argmax([acc_svm, acc_LinM, acc_ngb, acc_nBay])
classifiers = {0: 'SVM', 1: 'Linear Model - Perceptron', 2: 'K Nearest Neighbor', 3: 'Naive Bayes - GaussianNB'}
print('Best gender classifer is: {}'.format(classifiers[index]))

# Ran in Jupyter:- Prints SVM