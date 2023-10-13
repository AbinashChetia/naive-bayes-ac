import numpy as np
import pandas as pd

class NaiveBayesAC:
    def __init__(self):
        self.classes = None
        self.cls_prob = None
        self.likelihood = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.cls_prob = {}
        self.likelihood = {}

        for cls in self.classes:
            self.cls_prob[cls] = X[y == cls].shape[0] / X.shape[0]
            self.likelihood[cls] = {}
            
            for col in X.columns:
                self.likelihood[cls][col] = {}
                for val in np.unique(X[col]):
                    self.likelihood[cls][col][val] = (X[(X[col] == val) & (y == cls)].shape[0] + 1) / (X[y == cls].shape[0] + len(np.unique(X[col])))

    def predict(self, X):
        return [self._pred_row(X.iloc[i, :]) for i in range(X.shape[0])]
    
    def _pred_row(self, x):
        posteriors = []
        for cls in self.classes:
            likelihood = 1
            for col in x.index:
                likelihood *= self.likelihood[cls][col][x[col]]
            posteriors.append(self.cls_prob[cls] * likelihood)
        return self.classes[np.argmax(posteriors)]

