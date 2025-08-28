# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:25:31 2023

@author: shefai
"""
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes = (50, 20), alpha = 0.0001,
                 learning_rate = 'constant', learning_rate_init = 0.0001, max_iter = 5):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

    def fit(self, X, y):
        
        model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation='relu', 
                              solver='adam', max_iter=self.max_iter, alpha=self.alpha, learning_rate= self.learning_rate,
                              learning_rate_init=self.learning_rate_init)

        # Train the model
        model.fit(X, y)       
        self.model = model
        
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
        
    
    def clear(self):
        pass
    
    
    
    
    
    
    
    