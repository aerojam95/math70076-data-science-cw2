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
import numpy as np
from scipy.spatial.distance import cdist
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
        self.testData = torch.randn(1, 1, 28, 28)
        self.model = GraphNeuralNetwork(imageDimensions=28, numClasses=10, predEdge=False)
        
    def testInitialization(self):
        """
        Test the initialization of GraphNeuralNetwork
        
        Ensures that the layers that will be used in the model are initiliased correctly before the model is built
        """
        # Test the initialization of the GraphNeuralNetwork class
        self.assertFalse(self.model.predEdge)
        
        # Ensure that the adjacency matrix is initialized correctly
        self.assertTrue(hasattr(self.model, 'A'))  # Ensure A attribute exists
        self.assertEqual(self.model.A.shape, torch.Size([784, 784]))  # Ensure the shape of A is correct

        # Test the initialization when predEdge is True
        modelPredEdge = GraphNeuralNetwork(imageDimensions=28, numClasses=10, predEdge=True)
        self.assertTrue(hasattr(modelPredEdge, 'adjacencyMatrix'))  # Ensure adjacencyMatrix attribute exists
        self.assertTrue(hasattr(modelPredEdge, 'predEdgeFc'))
        
    def testForward(self):
        """
        Test the forward pass of the GraphNeuralNetwork

        Ensures that the output of the forward method has the correct shape given a batch of inputs,
        matching the expected batch size and number of class predictions
        """
        outputs = self.model(self.testData)
        self.assertEqual(outputs.shape, torch.Size([1, 10]))
        
    def testPrecomputeAdjacencyImages(self):
        # Test the precomputeAdjacencyImages method
        AHat = GraphNeuralNetwork.precomputeAdjacencyImages(imageDimensions=28, numClasses=10)
        self.assertEqual(AHat.shape, torch.Size([784, 784]))  # Ensure the output shape is as expected
        # Ensure the adjacency matrix is symmetric
        self.assertTrue(torch.allclose(AHat, AHat.T))
        # Ensure values are correctly computed
        col, row = np.meshgrid(np.arange(28), np.arange(28))
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / 28
        dist = cdist(coord, coord)
        sigma = 0.05 * np.pi
        expectedAHat = np.exp(-dist / sigma ** 2)
        expectedAHat[expectedAHat < 0.01] = 0
        expectedAHat = torch.from_numpy(expectedAHat).float()
        D = expectedAHat.sum(1)
        DHat = (D + 1e-5) ** (-0.5)
        expectedAHat = DHat.view(-1, 1) * expectedAHat.float() * DHat.view(1, -1)
        expectedAHat[expectedAHat > 0.0001] = expectedAHat[expectedAHat > 0.0001] - 0.2
        self.assertTrue(torch.allclose(AHat[:10, :10], expectedAHat[:10, :10]))
            
if __name__ == "__main__":
    unittest.main()