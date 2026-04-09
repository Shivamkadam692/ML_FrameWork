import numpy as np

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        # Handle zero variance
        self.var_[self.var_ == 0.0] = 1.0
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        if self.mean_ is None or self.var_ is None:
            raise ValueError("Scaler is not fitted yet.")
        return (X - self.mean_) / np.sqrt(self.var_)
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)
