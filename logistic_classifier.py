"""
Created on Tue Aug  6 09:06:05 2019
@author: mdrozdov
"""
### Logistic Classifier

from sklearn.metrics import accuracy_score
import sklearn.datasets
import numpy as np

class logistic_classifier(object):
    def __init__(self, lr = 0.01, num_iter = 10000, fit_intercept = True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept,X), axis = 1)

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1-y) * np.log(1-h))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h-y)) / y.size
            self.theta -= self.lr * gradient

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold


iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

model = logistic_classifier(lr = 0.1)
%time model.fit(X, y)

accuracy_score(y, model.predict(X, threshold = 0.5))
