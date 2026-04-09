class BaseModel:
    """Base class for all Machine Learning models."""
    def fit(self, X, y):
        raise NotImplementedError("The fit method must be implemented by subclasses.")
        
    def predict(self, X):
        raise NotImplementedError("The predict method must be implemented by subclasses.")
