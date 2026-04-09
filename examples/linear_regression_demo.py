import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.sample_dataset_loader import load_housing_dataset
from preprocessing.train_test_split import train_test_split_custom
from preprocessing.feature_scaling import StandardScaler
from models.linear_regression import LinearRegression
from metrics.mean_squared_error import mean_squared_error
from visualization.loss_curve import plot_loss_curve

def run():
    print("Loading California Housing Dataset...")
    X, y = load_housing_dataset()
    
    # Use a small subset to run quickly
    X, y = X[:1000], y[:1000]
    
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training Custom Linear Regression...")
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Test MSE: {mse:.4f}")
    
    print("Visualizing Training Loss...")
    plot_loss_curve(model.loss_history, title="Linear Regression Training Loss")

if __name__ == "__main__":
    run()
