import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, distance_metric='manhattan'):
        self.k = k
        self.distance_metric = distance_metric
        
    def _calculate_distance(self, X, x_test_item):
        if self.distance_metric == 'manhattan':
            return np.sum(np.abs(X - x_test_item), axis=1)
        elif self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X - x_test_item) ** 2, axis=1))
        else:
            raise ValueError("Метрика должна быть 'manhattan' или 'euclidean'")
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = []
        for x_test_item in X:
            distances = self._calculate_distance(self.X_train, x_test_item)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        probas = []
        for x_test_item in X:
            distances = self._calculate_distance(self.X_train, x_test_item)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            counts = np.bincount(k_nearest_labels, minlength=len(self.classes_))
            probas.append(counts / counts.sum())
            
        return np.array(probas)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

