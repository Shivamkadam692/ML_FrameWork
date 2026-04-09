import numpy as np

def r2_score(y_true, y_pred):
    """Calculate the R^2 (coefficient of determination) regression score function."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)
