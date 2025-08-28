# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF:
    def __init__(self, n_estimators = 30, max_depth = None, min_samples_split = 10, min_samples_leaf = 1):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, y):
        clf = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth,
                                     min_samples_split = self.min_samples_split,
                                     min_samples_leaf = self.min_samples_leaf)
        
        clf.fit(X, y)
        self.clf = clf
        
    def predict(self, X):
        y_predict = self.clf.predict(X)
        y_predict_prob = self.clf.predict_proba(X)
        return y_predict, y_predict_prob
    
    def clear(self):
        self.n_estimators = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    