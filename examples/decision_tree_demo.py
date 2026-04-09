import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.sample_dataset_loader import load_iris_dataset
from preprocessing.train_test_split import train_test_split_custom
from models.decision_tree import DecisionTree
from visualization.metrics_dashboard import print_metrics_dashboard

def run():
    print("Loading Iris Dataset...")
    X, y = load_iris_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    print("Training Custom Decision Tree...")
    model = DecisionTree(max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    predictions = model.predict(X_test)
    print_metrics_dashboard(y_test, predictions, title="Decision Tree on Iris")

if __name__ == "__main__":
    run()
