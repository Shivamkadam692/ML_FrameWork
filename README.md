# Machine Learning Framework (From Scratch)

A fully functional, educational Machine Learning Framework implemented from scratch using Python and NumPy. This project is built to understand the internal workings of machine learning algorithms.

## Project Structure
```text
ml_framework/
  core/                  # Core math and matrix operations
  preprocessing/         # Data loading, splitting, and scaling
  models/                # ML algorithms (Linear, Logistic, Decision Tree)
  optimizers/            # Gradient descent and learning rate schedulers
  metrics/               # Evaluation metrics
  visualization/         # Loss curves and confusion matrices
  datasets/              # Dataset loading utilities
  benchmark/             # Comparison with Scikit-learn
  examples/              # Example scripts running the models
  tests/                 # Unit tests
  main.py                # Main execution script
```

## Setup & Installation

Requirements:
- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn (strictly for benchmark comparison and fetching datasets)

To install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

You can run the framework using the `main.py` entry point. Ensure your terminal is in the project root directory.

### Running the Web UI

For a modern, interactive experience, you can run the web interface which provides dynamic training metrics and beautiful loss curve visualizations built with HTML, CSS, and Chart.js.

```bash
python app.py
```
Then, open your browser and navigate to `http://localhost:5000`.

### Running Demonstrations (CLI)

The examples will train the models from scratch and display performance metrics as well as visualizations (like loss curves and confusion matrices).

**Linear Regression** (on California Housing dataset):
```bash
python main.py demo-linear
```

**Logistic Regression** (on Breast Cancer dataset):
```bash
python main.py demo-logistic
```

**Decision Tree Classification** (on Iris dataset):
```bash
python main.py demo-tree
```

### Running Benchmarks
To compare the custom implementations against Scikit-Learn side-by-side with execution time and relative accuracy/MSE:
```bash
python main.py benchmark
```

### Running Unit Tests
Execute the tests directly using Python's unittest module:
```bash
python tests/unit_tests.py
```

## Implementation Details
- **Linear Regression**: Uses Mean Squared Error (MSE) and is optimized using batch Gradient Descent.
- **Logistic Regression**: Uses Binary Cross-Entropy Loss, the Sigmoid activation function, and is optimized using batch Gradient Descent.
- **Decision Tree**: A recursive binary splitting algorithm that uses Entropy and Information Gain to build the tree. Complete with leaf-node computation and max depth / min-samples stopping rules.
- **Optimizers**: A manual Gradient Descent implementation capable of incorporating learning rate decays (Step and Exponential).
- **Core Maths**: Softmax, cross entropy, dot product logic are mapped clearly against numpy operators mimicking mathematical formulas directly.

## Goal
The ultimate goal of this framework is educational transparency. No high-level modeling libraries were used to implement the algorithms, meaning every math operation and optimization step is fully visible in the source code. It demonstrates how back-computation forms weights in Linear settings, and recursively partitions data for tree nodes.
