import numpy as np
from .base_model import BaseModel
from optimizers.gradient_descent import GradientDescent

class LinearRegression(BaseModel):
    """Linear Regression model trained using Gradient Descent."""
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
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        optimizer = GradientDescent(learning_rate=self.learning_rate, n_iterations=self.n_iterations, tol=self.tol, lr_scheduler=self.lr_scheduler)
        self.weights, self.bias, self.loss_history = optimizer.optimize_linear_regression(X, y, self.weights, self.bias)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        return np.dot(X, self.weights) + self.bias
