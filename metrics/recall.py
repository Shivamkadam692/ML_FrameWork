import numpy as np

def recall_score(y_true, y_pred):
    """Calculate the recall score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    if true_positives + false_negatives == 0:
        return 0.0
        
    return true_positives / (true_positives + false_negatives)
