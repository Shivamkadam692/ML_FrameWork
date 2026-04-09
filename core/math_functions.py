import numpy as np

def sigmoid(z):
    """Compute the sigmoid function."""
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    """Compute the softmax function."""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def binary_cross_entropy(y_true, y_pred):
    """Compute Binary Cross Entropy Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)
