import pandas as pd
import numpy as np
from sklearn import datasets

def load_iris_dataset():
    """Load the Iris dataset."""
    data = datasets.load_iris()
    return data.data, data.target, data.feature_names

def load_breast_cancer_dataset():
    """Load the Breast Cancer dataset."""
    data = datasets.load_breast_cancer()
    return data.data, data.target, data.feature_names

def load_housing_dataset():
    """Load the California Housing dataset."""
    data = datasets.fetch_california_housing()
    return data.data, data.target, data.feature_names
