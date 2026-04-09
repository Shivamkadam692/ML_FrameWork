from .precision import precision_score
from .recall import recall_score

def f1_score(y_true, y_pred):
    """Calculate the F1 score."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)
