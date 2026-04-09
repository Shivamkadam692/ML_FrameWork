import numpy as np

def check_is_fitted(model, attributes):
    """Check if model has been strictly fitted."""
    for attr in attributes:
        if not hasattr(model, attr) or getattr(model, attr) is None:
            raise ValueError(f"This {type(model).__name__} instance is not fitted yet.")

def to_array(X):
    """Ensure X is a numpy array."""
    return np.asarray(X)
