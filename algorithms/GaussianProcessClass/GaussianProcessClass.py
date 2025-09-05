# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

import numpy as np

class GaussianProcessClass:
    def __init__(self, kernel = "DotProduct"):
        if kernel == "DotProduct":
            self.kernel = 1* DotProduct()
        elif kernel == "Matern":
            self.kernel = 1* Matern()
        elif kernel == "RationalQuadratic":
            self.kernel = 1* RationalQuadratic()
        elif  kernel == "WhiteKernel":
            self.kernel = 1* WhiteKernel()
        else: 
            self.kernel = 1* DotProduct()
        
    def fit(self, X,y):

        self.model = GaussianProcessClassifier(kernel = self.kernel, n_restarts_optimizer=10, random_state=42)
        
        self.model.fit(X, y)
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob

    def clear(self):
        self.max_depth = 0
        self.criterion = ""
        self.splitter = 0
    
    
    
    