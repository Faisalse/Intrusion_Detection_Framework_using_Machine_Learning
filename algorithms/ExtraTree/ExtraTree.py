# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

class ExtraTree:
    def __init__(self, n_estimators = 30, criterion = 'gini', max_depth = 10, 
                 min_samples_leaf = 5, max_features = "sqrt"):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
       
        
    def fit(self, train, test):
        clf = ExtraTreesClassifier(n_estimators = self.n_estimators, criterion = self.criterion,  
                            max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf,
                            max_features = self.max_features)
                                     
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
    
    
    
    