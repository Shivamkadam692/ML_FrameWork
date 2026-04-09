import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.sample_dataset_loader import load_breast_cancer_dataset
from preprocessing.train_test_split import train_test_split_custom
from preprocessing.feature_scaling import StandardScaler
from models.logistic_regression import LogisticRegression
from visualization.metrics_dashboard import print_metrics_dashboard
from visualization.confusion_matrix_plot import plot_confusion_matrix
from metrics.confusion_matrix import confusion_matrix
from visualization.loss_curve import plot_loss_curve

def run():
    print("Loading Breast Cancer Dataset...")
    X, y = load_breast_cancer_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training Custom Logistic Regression...")
    model = LogisticRegression(learning_rate=0.05, n_iterations=1000)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    predictions = model.predict(X_test)
    print_metrics_dashboard(y_test, predictions)
    
    print("Visualizing...")
    plot_loss_curve(model.loss_history, title="Logistic Regression Training Loss")
    cm, classes = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, classes, title="Logistic Regression Confusion Matrix")

if __name__ == "__main__":
    run()
