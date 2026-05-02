from flask import Flask, render_template, request, jsonify
from datasets.sample_dataset_loader import load_housing_dataset, load_breast_cancer_dataset, load_iris_dataset
from preprocessing.train_test_split import train_test_split_custom
from preprocessing.feature_scaling import StandardScaler
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTree
from metrics.mean_squared_error import mean_squared_error
from metrics.accuracy import accuracy_score
from metrics.confusion_matrix import confusion_matrix
from metrics.r2_score import r2_score
from metrics.precision import precision_score
from metrics.recall import recall_score
from metrics.f1_score import f1_score
import numpy as np
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
        dataset_name = data.get('dataset', 'housing')
        
        if dataset_name == 'housing':
            X, y, feature_names = load_housing_dataset()
        else:
            return jsonify({'success': False, 'error': 'Please select a Regression dataset for Linear Regression.'})

        # Subsample for speed
        X, y = X[:1000], y[:1000] 
        
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LinearRegression(learning_rate=lr, n_iterations=iterations)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Extract loss history safely
        loss_history = getattr(model, 'loss_history', [])
        
        # Compute residuals for diagnostics
        residuals = (y_test - predictions).tolist()
        
        return jsonify({
            'success': True,
            'metric': f"Test MSE: {mse:.4f}",
            'loss_history': loss_history,
            'y_test': y_test[:100].tolist(),
            'predictions': predictions[:100].tolist(),
            'weights': np.abs(model.weights).tolist(),
            'feature_names': list(feature_names) if feature_names is not None else [],
            'residuals': residuals[:100],
            'r2_score': round(float(r2), 4),
            'mse': round(float(mse), 4),
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/train/logistic', methods=['POST'])
def train_logistic():
    try:
        data = request.json
        lr = float(data.get('learningRate', 0.05))
        iterations = int(data.get('iterations', 1000))
        dataset_name = data.get('dataset', 'breast_cancer')
        
        if dataset_name == 'breast_cancer':
            X, y, feature_names = load_breast_cancer_dataset()
        else:
            return jsonify({'success': False, 'error': 'Logistic Regression currently only supports binary classification (Breast Cancer Dataset).'})

        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(learning_rate=lr, n_iterations=iterations)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        cm, classes = confusion_matrix(y_test, predictions)
        loss_history = getattr(model, 'loss_history', [])
        
        return jsonify({
            'success': True,
            'metric': f"Test Accuracy: {acc * 100:.2f}%",
            'loss_history': loss_history,
            'y_test': y_test[:100].tolist(),
            'predictions': predictions[:100].tolist(),
            'probabilities': probabilities[:100].tolist(),
            'weights': np.abs(model.weights).tolist(),
            'feature_names': list(feature_names) if feature_names is not None else [],
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/train/tree', methods=['POST'])
def train_tree():
    try:
        data = request.json
        max_depth = int(data.get('maxDepth', 5))
        min_samples = int(data.get('minSamples', 2))
        dataset_name = data.get('dataset', 'iris')
        
        if dataset_name == 'iris':
            X, y, feature_names = load_iris_dataset()
        elif dataset_name == 'breast_cancer':
            X, y, feature_names = load_breast_cancer_dataset()
        else:
            return jsonify({'success': False, 'error': 'Please select a Classification dataset for Decision Tree.'})

        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        cm, classes = confusion_matrix(y_test, predictions)
        
        # Feature importance from tree structure
        n_features = X_train.shape[1]
        importances = model.feature_importances(n_features)
        
        # Compute per-class metrics (using macro-average for multiclass)
        # For binary, use standard; for multiclass, compute macro average
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            prec = precision_score(y_test, predictions)
            rec = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
        else:
            # Macro-average for multiclass
            prec_sum, rec_sum, f1_sum = 0, 0, 0
            for cls in unique_classes:
                y_bin_true = (y_test == cls).astype(int)
                y_bin_pred = (predictions == cls).astype(int)
                p = precision_score(y_bin_true, y_bin_pred)
                r = recall_score(y_bin_true, y_bin_pred)
                f = f1_score(y_bin_true, y_bin_pred)
                prec_sum += p
                rec_sum += r
                f1_sum += f
            n_cls = len(unique_classes)
            prec = prec_sum / n_cls
            rec = rec_sum / n_cls
            f1 = f1_sum / n_cls

        return jsonify({
            'success': True,
            'metric': f"Test Accuracy: {acc * 100:.2f}%",
            'loss_history': [],  # Tree doesn't use gradient descent
            'y_test': y_test[:100].tolist(),
            'predictions': predictions[:100].tolist(),
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'weights': importances.tolist(),
            'feature_names': list(feature_names) if feature_names is not None else [],
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'max_depth_used': max_depth,
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
