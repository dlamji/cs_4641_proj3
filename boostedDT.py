'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
import copy

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.Hx = None
        self.betax = None
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
        n,d = X.shape
        weights = np.full((n),1./n)
        K = len(set(y))
        self.yClass = list(set(y))
        # only used for challenge question
        if(not 0 in self.yClass):
            self.yClass = [0] + self.yClass
        self.Hx = []
        self.betax = np.zeros((self.numBoostingIters))
        for t in range(self.numBoostingIters):
            errorsum = 0
            weightsum = 0
            clf = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth)
            clf.fit(X,y,sample_weight=weights)
            ypred = clf.predict(X)
            for i in range(n):
                errorsum += weights[i]*indicator(ypred[i],y[i])
            # errorsum /= weightsum
            beta = (np.log((1-errorsum)/errorsum) + np.log(K-1))/2
            weightsum = 0
            for i in range(n):
                weights[i] *= np.exp(beta*indicator(y[i],ypred[i]))
                weightsum += weights[i]
            weights /= weightsum
            self.betax[t] = beta
            self.Hx.append(copy.deepcopy(clf))


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        n,d = X.shape
        K = len(self.yClass)
        betas = np.zeros((n,K))
        ypred = np.zeros(n)
        # print self.yClass
        for i in range(self.numBoostingIters):
            tmppred = self.Hx[i].predict(X)
            for j in range(n):
                for yc in range(K):
                    betas[j][tmppred[j]] += self.betax[i]*indicator(yc,tmppred[j])
        for i in range(n):
            ypred[i] = np.argmax(betas[i])
        # print ypred.astype(int)
        return ypred

def indicator(a,b):
    if(a!=b):
        return 1
    return 0
