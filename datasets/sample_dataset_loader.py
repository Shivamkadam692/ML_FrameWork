import pandas as pd
import numpy as np
from sklearn import datasets

def load_iris_dataset():
    """Load the Iris dataset."""
    X, y = datasets.load_iris(return_X_y=True)
    return X, y

def load_breast_cancer_dataset():
    """Load the Breast Cancer dataset."""
    X, y = datasets.load_breast_cancer(return_X_y=True)
    return X, y

def load_housing_dataset():
    """Load the California Housing dataset."""
    X, y = datasets.fetch_california_housing(return_X_y=True)
    return X, y
