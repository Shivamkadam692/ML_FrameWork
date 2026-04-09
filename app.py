from flask import Flask, render_template, request, jsonify
from datasets.sample_dataset_loader import load_housing_dataset, load_breast_cancer_dataset, load_iris_dataset
from preprocessing.train_test_split import train_test_split_custom
from preprocessing.feature_scaling import StandardScaler
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTree
from metrics.mean_squared_error import mean_squared_error
from metrics.accuracy import accuracy_score
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train/linear', methods=['POST'])
def train_linear():
    try:
        data = request.json
        lr = float(data.get('learningRate', 0.01))
        iterations = int(data.get('iterations', 1000))
        
        # Load and sample dataset
        X, y = load_housing_dataset()
        X, y = X[:1000], y[:1000] # Subsample for speed
        
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LinearRegression(learning_rate=lr, n_iterations=iterations)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Extract loss history safely
        loss_history = getattr(model, 'loss_history', [])
        
        return jsonify({
            'success': True,
            'metric': f"Test MSE: {mse:.4f}",
            'loss_history': loss_history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/train/logistic', methods=['POST'])
def train_logistic():
    try:
        data = request.json
        lr = float(data.get('learningRate', 0.05))
        iterations = int(data.get('iterations', 1000))
        
        X, y = load_breast_cancer_dataset()
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(learning_rate=lr, n_iterations=iterations)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        loss_history = getattr(model, 'loss_history', [])
        
        return jsonify({
            'success': True,
            'metric': f"Test Accuracy: {acc * 100:.2f}%",
            'loss_history': loss_history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/train/tree', methods=['POST'])
def train_tree():
    try:
        data = request.json
        max_depth = int(data.get('maxDepth', 5))
        min_samples = int(data.get('minSamples', 2))
        
        X, y = load_iris_dataset()
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        return jsonify({
            'success': True,
            'metric': f"Test Accuracy: {acc * 100:.2f}%",
            'loss_history': [] # Tree doesn't use gradient descent
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
