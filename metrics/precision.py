import numpy as np

def precision_score(y_true, y_pred):
    """Calculate the precision score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    if true_positives + false_positives == 0:
        return 0.0
        
    return true_positives / (true_positives + false_positives)
