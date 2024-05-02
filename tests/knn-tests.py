#=============================================================================
# Unit test:
# k-Nearest neighbours class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import sys
import os
from sklearn.datasets import make_classification

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from knn import kNearestNeighbours

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class from knn.py
#=============================================================================

class TestkNearestNeighbours(unittest.TestCase):
    """
    Unit tests for the kNearestNeighbours class

    This test class provides a series of automated tests to validate the behavior and functionality of the
    kNearestNeighbours class, including its ability to train, run predictions, save to and load from files
    """
    def setUp(self):
        """
        Set up method to prepare a test environment before each test

        Initializes a kNearestNeighbours object with a fixed smoothing parameter and sets up a simple,
        synthetic dataset for testing
        """
        self.trainImages, self.trainLabels = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=0, random_state=42)
        self.testImages, self.testLabels = make_classification(n_samples=10, n_features=20, n_informative=2, n_redundant=0, random_state=42)
        self.model = kNearestNeighbours(k=3, nSplits=5)
        self.testPath = os.getcwd() + "/"
        self.model.train(self.testPath, self.trainImages, self.trainLabels, kmin=1, kmax=3)

    def testInitialization(self):
        """
        Test initialization of kNearestNeighbours class
        
        Validates that the method initialises model correctly
        """
        self.assertEqual(self.model.k, 3)
        self.assertEqual(self.model.nSplits, 5)
        self.assertIsNotNone(self.model)

    def testTrain(self):
        """
        Test the train method of the kNearestNeighbours class

        Validates that the method executes training and generates model parameters
        """
        file = f"{self.testPath}knn_training_curve.png"
        self.assertTrue(os.path.exists(file))
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")
            
    def testEvaluate(self):
        """
        Test evaluation functionality of kNearestNeighbours model
        
        Validates that the method executes and evaluates the model
        """
        accuracy = self.model.evaluate(self.testImages, self.testLabels)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
    def testPredict(self):
        """
        Test prediction functionality of kNearestNeighbours model
        
        Validates that the method executes and predicts using the model
        """
        predictions = self.model.predict(self.testImages)
        self.assertEqual(len(predictions), len(self.testLabels))

    def testSaveModel(self):
        """
        Test the saveModel method of the kNearestNeighbours class

        Checks if the model can be saved without errors. This test ensures that file operations within
        the saveModel method are functioning correctly
        """
        # Assuming saveModel simply writes to a file, this could be mocked or checked if a file exists
        file = "test_knn_model_parameters.txt"
        self.model.saveModel(file)
        # Normally, you"d use mock here or check if file exists, then clean up
        self.assertTrue(os.path.exists(file))
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")
            
    def testLoadModel(self):
        """
        Test the loadModel method of the kNearestNeighbours class

        Verifies that model parameters are correctly restored from a file and that the loaded model
        matches the original model in terms of parameter values
        """
        file = "test_knn_model_parameters.txt"
        self.model.saveModel(file)
        loadedModel = kNearestNeighbours(k=3, nSplits=5)
        loadedModel.loadModel(file)
        self.assertEqual(self.model.k, loadedModel.k)
        self.assertEqual(self.model.nSplits, loadedModel.nSplits)
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

if __name__ == "__main__":
    unittest.main()