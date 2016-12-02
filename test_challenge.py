"""
======================================================
Test the boostedDT against the standard decision tree
======================================================
Author: Eric Eaton, 2014
"""
print(__doc__)
from numpy import loadtxt, ones, zeros, where
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys, traceback
from boostedDT import BoostedDT

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold


# load the data set
filename = 'data/challengeTrainLabeled.dat'
data = loadtxt(filename, delimiter=',')

X = data[:, 0:10]
y = data[:, 10]


n,d = X.shape
nTrain = 0.5*n

kf = KFold(n,10) 

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

knnmean = []
knnstd = []
for knei in range(2,20):
	accukf = []
	for train_index,test_index in kf:
		# print("TRAIN:", train_index, "TEST:", test_index)
		Xtrain, Xtest = X[train_index], X[test_index]
		ytrain, ytest = y[train_index], y[test_index]
		# train the decision tree
		modelDT = DecisionTreeClassifier()
		modelDT.fit(Xtrain,ytrain)
		# train the KNN
		modelKNN = KNeighborsClassifier(n_neighbors=knei)
		modelKNN.fit(Xtrain,ytrain)
		# output predictions on the remaining data
		ypred_DT = modelDT.predict(Xtest)
		# ypred_BoostedDT = modelBoostedDT.predict(Xtest)
		ypred_KNN = modelKNN.predict(Xtest)

		accuracyDT = accuracy_score(ytest, ypred_DT)
		# accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)
		accuracyKNN = accuracy_score(ytest,ypred_KNN)
		# print "KNN Accuracy = "+str(accuracyKNN)
		accukf.append(accuracyKNN)
	tmp = np.array(accukf)
	print np.mean(tmp)
	knnmean.append(np.mean(tmp))
	knnstd.append(np.std(tmp))

print min(knnstd)
print knnstd.index(min(knnstd))




# split the data

# Xtrain = X[:nTrain,:]
# ytrain = y[:nTrain]
# Xtest = X[nTrain:,:]
# ytest = y[nTrain:]



# train the boosted DT
# modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=3)
# modelBoostedDT.fit(Xtrain,ytrain)

# only for the chosen best classifier
filename = 'data/challengeTestUnlabeled.dat'
data = loadtxt(filename, delimiter=',')

Xtest_unlabeled = data[:,0:10]
ypred_unlabeled = modelKNN.predict(Xtest_unlabeled)

# print ypred_BoostedDT

# print ','.join(str(e) for e in ypred_unlabeled.astype(int))
# print "Decision Tree Accuracy = "+str(accuracyDT)
# print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)
# print "KNN Accuracy = "+str(accuracyKNN)