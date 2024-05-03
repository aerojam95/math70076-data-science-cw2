#=============================================================================
# Unit test:
# Convolutional neural network class unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import torch
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from cnn import ConvolutionalNeuralNetwork

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
        # Initialize the neural network
        self.image_dimension = 28
        self.model = ConvolutionalNeuralNetwork(imageDimensions=self.image_dimension)
        
    def testInitialization(self):
        """
        Test the initialization of ConvolutionalNeuralNetwork
        
        Ensures that the layers that will be used in the model are initiliased correctly before the model is built
        """
        self.assertIsInstance(self.model.conv1, Conv2d, "Conv Layer 1 should be an instance of torch.nn.Conv2d")
        self.assertIsInstance(self.model.conv2, Conv2d, "Conv Layer 2 should be an instance of torch.nn.Conv2d")
        self.assertIsInstance(self.model.conv3, Conv2d, "Conv Layer 3 should be an instance of torch.nn.Conv2d")
        self.assertIsInstance(self.model.conv4, Conv2d, "Conv Layer 4 should be an instance of torch.nn.Conv2d")
        self.assertIsInstance(self.model.fc1, Linear, "FC Layer 1 should be an instance of torch.nn.Linear")
        self.assertIsInstance(self.model.fc2, Linear, "FC Layer 2 should be an instance of torch.nn.Linear")
        self.assertIsInstance(self.model.fc3, Linear, "FC Layer 3 should be an instance of torch.nn.Linear")
        self.assertIsInstance(self.model.activation, ReLU, "Activation should be an instance of torch.nn.ReLU")
        self.assertIsInstance(self.model.pool, MaxPool2d, "Pooling should be an instance of torch.nn.MaxPool2d")
        self.assertTrue(callable(self.model.softmax))

        # Check image dimensions
        self.assertEqual(self.model.imageDimensions, 28, "Image dimensions mismatch")
        
        # Check convolutional layers
        conv_layers = [
            (self.model.conv1, 1, 32, 3),
            (self.model.conv2, 32, 64, 3),
            (self.model.conv3, 64, 128, 3),
            (self.model.conv4, 128, 128, 1)
        ]
        for i, (layer, in_channels, out_channels, kernel_size) in enumerate(conv_layers, 1):
            with self.subTest(layer=f"conv{i}"):
                self.assertIsInstance(layer, Conv2d, f"conv{i} should be an instance of nn.Conv2d")
                self.assertEqual(layer.in_channels, in_channels, f"Incorrect number of input channels in conv{i}")
                self.assertEqual(layer.out_channels, out_channels, f"Incorrect number of output channels in conv{i}")
                self.assertEqual(layer.kernel_size, (kernel_size, kernel_size), f"Incorrect kernel size in conv{i}")

        # Check linear layers
        linear_layers = [
            (self.model.fc1, 128, 64),
            (self.model.fc2, 64, 32),
            (self.model.fc3, 32, 10)
        ]
        for i, (layer, in_features, out_features) in enumerate(linear_layers, 1):
            with self.subTest(layer=f"fc{i}"):
                self.assertIsInstance(layer, Linear, f"FC{i} should be an instance of nn.Linear")
                self.assertEqual(layer.in_features, in_features, f"Incorrect input features in FC{i}")
                self.assertEqual(layer.out_features, out_features, f"Incorrect output features in FC{i}")
        

    def testForward(self):
        """
        Test the forward pass of the ConvolutionalNeuralNetwork

        Ensures that the output of the forward method has the correct shape given a batch of inputs,
        matching the expected batch size and number of class predictions
        """
        input_tensor = torch.randn(1, 1, self.image_dimension, self.image_dimension)
        output = self.model.forward(input_tensor)
        self.assertEqual(output.shape, (1, 10), "Output tensor should be of shape (1, number of classes)")
        self.assertTrue(torch.allclose(output.sum(), torch.tensor(1.0)), "Output probabilities should sum to 1")
        
if __name__ == "__main__":
    unittest.main()