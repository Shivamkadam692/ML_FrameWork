import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)
