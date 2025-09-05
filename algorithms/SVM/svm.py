import numpy as np
from sklearn import svm

class SVM:
    """
    Thin wrapper around sklearn.svm.SVC for binary & multiclass classification.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 probability=True, decision_function_shape='ovr',
                 class_weight=None, cache_size=200):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.probability = bool(probability)
        self.decision_function_shape = decision_function_shape
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.model = None

    def fit(self, X, y):
        # SVC handles multiclass natively; just choose decision_function_shape
        self.model = svm.SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            decision_function_shape=self.decision_function_shape,
            class_weight=self.class_weight,
            cache_size=self.cache_size
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if not self.probability:
            raise RuntimeError("probability=False. Re-init with probability=True.")
        return self.model.predict_proba(X)

    def clear(self):
        """Reset to defaults and drop the trained model."""
        self.C = 1.0
        self.kernel = 'rbf'
        self.gamma = 'scale'
        self.probability = False
        self.decision_function_shape = 'ovr'
        self.class_weight = None
        self.cache_size = 200
        self.model = None
