'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.fitMatrix = None
        self.priorY = None
        self.K = 10
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        self.priorY = np.zeros((10))
        # the probability table for y classes
        for i in xrange(n):
        	self.priorY[y[i]] += 1
        self.priorY /= n
        # for each attribute create a table
        self.fitMatrix = np.zeros((d,10,20))
        # loop for each attribute
        for i in range(d):
        	tmpTable = np.zeros((10,20))
        	colX = X[:,i]
        	xClass = self.countClass(colX)
        	# print xClass
        	# loop for each y class
        	for yClass in range(10):
        		lower = 0
        		# loop for every x example    	
        		for j in range(n):
        			if(y[j]==yClass):
        				lower += 1
        				tmpTable[yClass][colX[j]] += 1
        		# print tmpTable
	        	if(self.useLaplaceSmoothing):
        			tmpTable[yClass] += 1
        			lower += xClass
	        	tmpTable[yClass] /= lower
        	self.fitMatrix[i] = np.copy(tmpTable)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        Kprob = self.predictProbs(X)
        yPred = np.zeros((n))
        for i in xrange(n):
        	# argmax returns the index, which is what we need
        	yPred[i] = np.argmax(Kprob[i])
        # print yPred
        return yPred


    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n,d = X.shape
        Kprob = np.zeros((n,self.K))
        for i in xrange(n):
        	for yClass in range(self.K):
        		tmpProb = np.log(self.priorY[yClass])
        		for attr in range(d):
        			tmpProb += np.log(self.fitMatrix[attr][yClass][X[i][attr]])
        		Kprob[i][yClass] = tmpProb
        return Kprob

    def countClass(self,X):
    	return len(set(X))
        
        