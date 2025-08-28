# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators = 30, learning_rate = 0.1, depth = 4, l2_leaf_reg = 1):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
       
        
    def fit(self, train, test):
        clf = AdaBoostClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate)
        
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
    
    
    
    