import numpy as np

class MinMaxScaler:
    """Normalize features by scaling each feature to a given range, typically [0, 1]."""
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.scale_ = self.data_max_ - self.data_min_
        # Handle zero variance
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.scale_ is None:
            raise ValueError("Scaler is not fitted yet.")
        X_std = (X - self.data_min_) / self.scale_
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)
