import matplotlib.pyplot as plt
from metrics.accuracy import accuracy_score
from metrics.precision import precision_score
from metrics.recall import recall_score
from metrics.f1_score import f1_score

def print_metrics_dashboard(y_true, y_pred, title="Classification Metrics Dashboard"):
    """Print a summary dashboard of classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("="*40)
    print(f" {title}")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*40)
