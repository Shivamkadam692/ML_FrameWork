import numpy as np

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
