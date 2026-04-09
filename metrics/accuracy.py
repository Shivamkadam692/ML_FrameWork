import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)
