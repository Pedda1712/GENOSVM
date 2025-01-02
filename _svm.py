from ._solver import solve
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class GENOSVM(ClassifierMixin, BaseEstimator):

    def __init__(self, C=1.0, max_iter=200, gpu=False):
        self.C = C
        self.max_iter = max_iter
        self.gpu = gpu
        
    def _more_tags(self):
        return {
            "binary_only": True,
            "poor_score": True,
            "pairwise": True
        }
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = True
        return tags


    def fit(self, K, y):
        self.classes = np.unique(y)
        self.y = np.where(y == self.classes[0], 1, -1)
        _, self.a = solve(K, self.C, self.y, np, self.max_iter, False)
        b_values = -self.y + np.sum(self.a * self.y * K, axis=1)
        self.b = np.median(b_values[np.where(self.a > 0)])
        return self

        
    def decision_function(self, K):
        y_predict = np.dot(self.a * self.y,K.T) - self.b
        return -y_predict

    def predict(self, K):
        return np.where(self.decision_function(K) < 0, self.classes[0], self.classes[min(1, len(self.classes) - 1)])
