#=============================================================================
# Unit test:
# Gaussian Naive Bayes class unit tests
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

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Custom modules
from gnb import GaussianNaiveBayes

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class for gnb.py
#=============================================================================

class TestGaussianNaiveBayes(unittest.TestCase):
    def setUp(self):
        # Initialize the GaussianNaiveBayes object
        self.model = GaussianNaiveBayes(smoothing=1000.0)
        # Create simple synthetic data for testing
        self.trainImages = np.array([[0, 0], [1, 1], [2, 2]])
        self.trainLabels = np.array([0, 1, 1])
        self.valImages = np.array([[0, 0], [2, 2]])
        self.valLabels = np.array([0, 1])

    def test_run(self):
        # Run the model on the test data
        self.model.run(self.trainImages, self.trainLabels, self.valImages, self.valLabels)
        # Test if predictions are correctly calculated
        self.assertEqual(len(self.model.pred), len(self.valLabels))
        # Check for valid probability estimates in likelihoods
        for ll in self.model.likelihood:
            flattened_probs = np.hstack(ll)
            self.assertTrue(np.all(flattened_probs >= 0))

    def test_accuracy(self):
        # Test the accuracy calculation
        self.model.run(self.trainImages, self.trainLabels, self.valImages, self.valLabels)
        expected_accuracy = 1.0
        self.assertEqual(self.model.accuracy, expected_accuracy)

    def test_save_model(self):
        # Test save model by checking if it runs without errors (actual file writing could be mocked)
        self.model.run(self.trainImages, self.trainLabels, self.valImages, self.valLabels)
        self.model.saveModel('test_model_save.txt')

    def test_save_validation(self):
        # Test save validation by checking if it runs without errors
        self.model.run(self.trainImages, self.trainLabels, self.valImages, self.valLabels)
        self.model.saveValidation('test_validation_save.txt')

if __name__ == '__main__':
    unittest.main()