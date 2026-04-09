import time
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import LogisticRegression as SklearnLogR
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.metrics import accuracy_score as sklearn_acc, mean_squared_error as sklearn_mse

from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTree
from metrics.accuracy import accuracy_score
from metrics.mean_squared_error import mean_squared_error

def benchmark_linear_regression(X_train, X_test, y_train, y_test):
    print("--- Linear Regression Benchmark ---")
    
    # Custom Framework
    start_time = time.time()
    custom_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    custom_time = time.time() - start_time
    custom_mse = mean_squared_error(y_test, custom_preds)
    
    print(f"Custom Framework -> MSE: {custom_mse:.4f}, Time: {custom_time:.4f}s")
    
    # Scikit-learn
    start_time = time.time()
    sk_model = SklearnLR()
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    sk_mse = sklearn_mse(y_test, sk_preds)
    
    print(f"Scikit-Learn     -> MSE: {sk_mse:.4f}, Time: {sk_time:.4f}s")
    print()

def benchmark_logistic_regression(X_train, X_test, y_train, y_test):
    print("--- Logistic Regression Benchmark ---")
    
    # Custom Framework
    start_time = time.time()
    custom_model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    custom_time = time.time() - start_time
    custom_acc = accuracy_score(y_test, custom_preds)
    
    print(f"Custom Framework -> Accuracy: {custom_acc:.4f}, Time: {custom_time:.4f}s")
    
    # Scikit-learn
    start_time = time.time()
    sk_model = SklearnLogR(max_iter=1000)
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    sk_acc = sklearn_acc(y_test, sk_preds)
    
    print(f"Scikit-Learn     -> Accuracy: {sk_acc:.4f}, Time: {sk_time:.4f}s")
    print()

def benchmark_decision_tree(X_train, X_test, y_train, y_test):
    print("--- Decision Tree Benchmark ---")
    
    # Custom Framework
    start_time = time.time()
    custom_model = DecisionTree(max_depth=5)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    custom_time = time.time() - start_time
    custom_acc = accuracy_score(y_test, custom_preds)
    
    print(f"Custom Framework -> Accuracy: {custom_acc:.4f}, Time: {custom_time:.4f}s")
    
    # Scikit-learn
    start_time = time.time()
    sk_model = SklearnDT(max_depth=5)
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    sk_acc = sklearn_acc(y_test, sk_preds)
    
    print(f"Scikit-Learn     -> Accuracy: {sk_acc:.4f}, Time: {sk_time:.4f}s")
    print()
