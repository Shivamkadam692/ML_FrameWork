import numpy as np

def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix to evaluate the accuracy of a classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1
        
    return matrix, classes
