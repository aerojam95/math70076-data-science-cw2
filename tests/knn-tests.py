#=============================================================================
# Unit test:
# k-Nearest neighbours class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
from unittest.mock import patch, mock_open
from io import StringIO
import numpy as np
import sys
import os
from sklearn.datasets import make_classification

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Custom modules
from knn import kNearestNeighbours

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class for knn.py
#=============================================================================

class TestkNearestNeighbours(unittest.TestCase):
    def setUp(self):
        """ Create a small synthetic dataset for testing """
        self.trainImages, self.trainLabels = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=0, random_state=42)
        self.testImages, self.testLabels = make_classification(n_samples=10, n_features=20, n_informative=2, n_redundant=0, random_state=42)
        self.model = kNearestNeighbours(k=3, nSplits=5)
        self.testPath = os.getcwd() + '/'

    def test_initialization(self):
        """ Test the initialization of kNearestNeighbours class """
        self.assertEqual(self.model.k, 3)
        self.assertEqual(self.model.nSplits, 5)
        self.assertIsNone(self.model.predictAccuracy)

    def test_train(self):
        """ Test the training method """
        accuracies = self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=1, kmax=5)
        self.assertIsInstance(accuracies, list)
        self.assertEqual(len(accuracies), 5)
        try:
            os.remove("knn_training_curve.png")
            print(f"Deleted training curve after test.")
        except OSError as e:
            print(f"Error: {e.strerror}")
        

    def test_predict(self):
        """ Test the prediction method """
        self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=3, kmax=3)
        predictions = self.model.predict(self.testImages)
        self.assertEqual(len(predictions), len(self.testLabels))

    def test_evaluate(self):
        """ Test the evaluation method """
        self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=3, kmax=3)
        accuracy = self.model.evaluate(self.testImages, self.testLabels)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_save_model(self):
        """ Test the save model method """
        # Assuming saveModel simply writes to a file, this could be mocked or checked if a file exists
        self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=3, kmax=3)
        filename = 'test_knn_model_parameters.txt'
        self.model.saveModel(filename)
        # Normally, you'd use mock here or check if file exists, then clean up
        self.assertTrue(os.path.exists(filename))
        try:
            os.remove(filename)
            print(f"Deleted {filename} after test.")
        except OSError as e:
            print(f"Error: {e.strerror}")

    def test_save_validation(self):
        """ Test the save validation method """
        self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=3, kmax=3)
        self.model.evaluate(self.testImages, self.testLabels)
        filename = 'test_knn_validation_results.txt'
        self.model.saveValidation(filename)
        self.assertTrue(os.path.exists(filename))
        try:
            os.remove(filename)
            print(f"Deleted {filename} after test.")
        except OSError as e:
            print(f"Error: {e.strerror}")

if __name__ == '__main__':
    unittest.main()