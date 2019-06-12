#SCIKIT-LEARN CLASSIFIERS
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


from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

clf = tree.DecisionTreeClassifier()
log_reg = LogisticRegression(penalty="12")
gnb = GaussionaNB()
knn = KNeighborsClassifier(n_neighbors=3) 

# CHALLENGE - create 3 more classifiers...
# 1 - KNeighborsClassifier(3)
# 2 - GaussianNB()
# 3 - LogisticRegression()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

log_reg = log_reg.fit(X,Y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = log_reg.predict_proba(X_new)

gnb = gnb.fit(X,Y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
Y_proba = gnb.predict_proba(X_new)
pred = gnb.predict(X)

knn = knn.fit(X,Y)

prediction1 = clf.predict([[190, 70, 43]])
prediction2 = log_reg.predict([[190, 70, 43]])
prediction3 = gnb.predict([[190, 70, 43]])
prediction4 = knn.predict([[190, 70, 43]])

# CHALLENGE compare their results and print the best one!

print(prediction1)
print(prediction2)
print(prediction3)
print(prediction4)