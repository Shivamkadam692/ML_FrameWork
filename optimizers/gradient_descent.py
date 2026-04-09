import numpy as np
from core.math_functions import sigmoid, binary_cross_entropy, mean_squared_error

class GradientDescent:
    """Optimization algorithm to find parameters minimizing the cost function."""
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6, lr_scheduler=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.lr_scheduler = lr_scheduler
        
    def optimize_linear_regression(self, X, y, weights, bias):
        n_samples = X.shape[0]
        loss_history = []
        
        for i in range(self.n_iterations):
            # Predictions
            y_predicted = np.dot(X, weights) + bias
            
            # Loss
            loss = mean_squared_error(y, y_predicted)
            loss_history.append(loss)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights
            lr = self.learning_rate if self.lr_scheduler is None else self.lr_scheduler(i, self.learning_rate)
            weights -= lr * dw
            bias -= lr * db
            
            # Convergence check
            if i > 0 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                break
                
        return weights, bias, loss_history

    def optimize_logistic_regression(self, X, y, weights, bias):
        n_samples = X.shape[0]
        loss_history = []
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, weights) + bias
            y_predicted = sigmoid(linear_model)
            
            loss = binary_cross_entropy(y, y_predicted)
            loss_history.append(loss)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            lr = self.learning_rate if self.lr_scheduler is None else self.lr_scheduler(i, self.learning_rate)
            weights -= lr * dw
            bias -= lr * db
            
            if i > 0 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                break
                
        return weights, bias, loss_history
