#=============================================================================
# Unit test:
# Gaussian Naive Bayes class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import numpy as np
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from gnb import GaussianNaiveBayes

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class from gnb.py
#=============================================================================

class TestGaussianNaiveBayes(unittest.TestCase):
    """
    Unit tests for the GaussianNaiveBayes class

    This test class provides a series of automated tests to validate the behavior and functionality of the
    GaussianNaiveBayes class, including its ability to train, run predictions, save to and load from files
    """
    def setUp(self):
        """
        Set up method to prepare a test environment before each test

        Initializes a GaussianNaiveBayes object with a fixed smoothing parameter and sets up a simple,
        synthetic dataset for testing
        """
        # Initialize the GaussianNaiveBayes object
        self.model = GaussianNaiveBayes(smoothing=1000.0)
        # Create simple synthetic data for testing
        self.trainImages = np.array([[0, 0], [1, 1], [2, 2]])
        self.trainLabels = np.array([0, 1, 1])
        self.valImages = np.array([[0, 0], [2, 2]])
        self.valLabels = np.array([0, 1])
        # Run the model on the synthetic test data
        self.model.train(self.trainImages, self.trainLabels)
        
    def testInitialization(self):
        """
        Test initialization of GaussianNaiveBayes class
        
        Validates that the method initialises model correctly
        """
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.smoothing, 1000.0)

    def testTrain(self):
        """
        Test the train method of the GaussianNaiveBayes class

        Validates that the method executes training and generates model parameters
        """
        self.assertEqual(len(self.model.classes), len(np.unique(self.trainLabels)))
        self.assertEqual(len(self.model.mean), len(np.unique(self.trainLabels)))
        self.assertEqual(len(self.model.standardDeviation), len(np.unique(self.trainLabels)))
        self.assertEqual(len(self.model.count), len(np.unique(self.trainLabels)))
        self.assertEqual(len(self.model.prior), len(np.unique(self.trainLabels)))

    def testEvaluate(self):
        """
        Test evaluation functionality of GaussianNaiveBayes model
        
        Validates that the method executes and evaluates the model
        """
        accuracy = self.model.evaluate(self.trainImages, self.trainLabels)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def testPredict(self):
        """
        Test prediction functionality of GaussianNaiveBayes model
        
        Validates that the method executes and predicts using the model
        """
        predictions = self.model.predict(self.valImages)
        self.assertEqual(len(predictions), len(self.valImages))

    def testSaveModel(self):
        """
        Test the saveModel method of the GaussianNaiveBayes class

        Checks if the model can be saved without errors. This test ensures that file operations within
        the saveModel method are functioning correctly
        """
        file = "test_gnb_model_parameters.txt"
        self.model.saveModel(file)
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")
            
    def testLoadModel(self):
        """
        Test the loadModel method of the GaussianNaiveBayes class

        Verifies that model parameters are correctly restored from a file and that the loaded model
        matches the original model in terms of parameter values
        """
        file = "test_gnb_model_parameters.txt"
        self.model.saveModel(file)
        loadedModel = GaussianNaiveBayes()
        loadedModel.loadModel(file)
        self.assertEqual(self.model.smoothing, loadedModel.smoothing)
        np.testing.assert_array_equal(self.model.classes, loadedModel.classes)
        np.testing.assert_array_equal(self.model.prior, loadedModel.prior)
        np.testing.assert_array_equal(self.model.mean, loadedModel.mean)
        np.testing.assert_array_equal(self.model.standardDeviation, loadedModel.standardDeviation)
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

if __name__ == "__main__":
    unittest.main()