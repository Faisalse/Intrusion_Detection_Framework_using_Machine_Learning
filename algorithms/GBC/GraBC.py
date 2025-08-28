# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class GBC:
    def __init__(self, n_estimators = 30, learning_rate = 0.1, max_depth = None, min_samples_split = 10, min_samples_leaf = 1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, train, test):
        clf = GradientBoostingClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  max_depth = self.max_depth,
                                     min_samples_split = self.min_samples_split,
                                     min_samples_leaf = self.min_samples_leaf)
        
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    