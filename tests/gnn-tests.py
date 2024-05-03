#=============================================================================
# Unit test:
# Graph Neural network class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import torch
from  torch.nn import CrossEntropyLoss, Linear, Flatten, ReLU
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from gnn import GraphNeuralNetwork

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class from nn.py
#=============================================================================

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up for the test case
        
        This method is called before each test function to set up any objects that may be needed for testing
        Initializes a NeuralNetwork instance and creates a DataLoader with synthetic data to be used in the tests
        """
        # Initialize the graph neural network
        self.image_dimension = 28
        self.model = GraphNeuralNetwork(imageDimensions=self.image_dimension)
        
    def testInitialization(self):
        """
        Test the initialization of GraphNeuralNetwork
        
        Ensures that the layers that will be used in the model are initiliased correctly before the model is built
        """
        

        # Check layer dimensions
        self.assertEqual(self.model.imageDimensions, 28, "Image dimensions mismatch")

    def testForward(self):
        """
        Test the forward pass of the GraphNeuralNetwork

        Ensures that the output of the forward method has the correct shape given a batch of inputs,
        matching the expected batch size and number of class predictions
        """
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, (10, 10))
        
if __name__ == "__main__":
    unittest.main()