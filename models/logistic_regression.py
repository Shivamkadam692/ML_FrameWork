import numpy as np
from .base_model import BaseModel
from optimizers.gradient_descent import GradientDescent
from core.math_functions import sigmoid

class LogisticRegression(BaseModel):
    """Logistic Regression model for binary classification."""
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6, lr_scheduler=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.lr_scheduler = lr_scheduler
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        optimizer = GradientDescent(learning_rate=self.learning_rate, n_iterations=self.n_iterations, tol=self.tol, lr_scheduler=self.lr_scheduler)
        self.weights, self.bias, self.loss_history = optimizer.optimize_logistic_regression(X, y, self.weights, self.bias)
        return self
        
    def predict_proba(self, X):
        X = np.asarray(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)
