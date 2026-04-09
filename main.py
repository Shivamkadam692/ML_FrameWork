import sys
import argparse
from examples.linear_regression_demo import run as run_linear
from examples.logistic_regression_demo import run as run_logistic
from examples.decision_tree_demo import run as run_tree
from benchmark.compare_with_sklearn import benchmark_linear_regression, benchmark_logistic_regression, benchmark_decision_tree
from datasets.sample_dataset_loader import load_housing_dataset, load_breast_cancer_dataset, load_iris_dataset
from preprocessing.train_test_split import train_test_split_custom
from preprocessing.feature_scaling import StandardScaler

def run_benchmarks():
    print("Running benchmarks against Scikit-Learn...")
    
    print("\n--- Preparing Data for Linear Regression ---")
    X, y = load_housing_dataset()
    X, y = X[:1000], y[:1000] # Subsample for speed
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    benchmark_linear_regression(X_train, X_test, y_train, y_test)
    
    print("\n--- Preparing Data for Logistic Regression ---")
    X, y = load_breast_cancer_dataset()
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    benchmark_logistic_regression(X_train, X_test, y_train, y_test)
    
    print("\n--- Preparing Data for Decision Tree ---")
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
    benchmark_decision_tree(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Framework")
    parser.add_argument('action', choices=['demo-linear', 'demo-logistic', 'demo-tree', 'benchmark'], 
                        help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == 'demo-linear':
        run_linear()
    elif args.action == 'demo-logistic':
        run_logistic()
    elif args.action == 'demo-tree':
        run_tree()
    elif args.action == 'benchmark':
        run_benchmarks()
