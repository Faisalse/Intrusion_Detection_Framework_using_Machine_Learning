# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 09:19:53 2023

@author: shefai
"""

from sklearn import svm
import numpy as np

class SVM:
    def __init__(self, C = 0.1, gamma = 1, kernel= 'linear'):
        self.C = C,
        self.gamma = gamma
        self.kernel = kernel
        
        
    def fit(self, train, test):
        clf = clf = svm.SVC(kernel = self.kernel, probability= True)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    def clear(self):
        self.C = 0
        self.gamma =1
        self.kernel = 0
        