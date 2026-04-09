import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

class TestMathFunctions(unittest.TestCase):
    def test_sigmoid(self):
        from core.math_functions import sigmoid
        z = np.array([0, 2, -2])
        res = sigmoid(z)
        self.assertAlmostEqual(res[0], 0.5)
        self.assertTrue(res[1] > 0.5)
        self.assertTrue(res[2] < 0.5)
        
    def test_matrix_multiplication(self):
        from core.matrix_operations import matrix_multiply
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        res = matrix_multiply(A, B)
        self.assertTrue(np.array_equal(res, np.array([[4, 4], [10, 8]])))

class TestMetrics(unittest.TestCase):
    def test_accuracy_score(self):
        from metrics.accuracy import accuracy_score
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 0.8)

if __name__ == '__main__':
    unittest.main()
