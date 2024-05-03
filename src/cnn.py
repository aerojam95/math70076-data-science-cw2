#=============================================================================
# Programme: 
# Class for Convolution neural network model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import torch
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.functional import softmax

# Custom modules
from nn import NeuralNetwork

#=============================================================================
# Functions
#=============================================================================
    
#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Classes
#=============================================================================

class ConvolutionalNeuralNetwork(NeuralNetwork):
    """
    A feedforward neural network for image classification

    Attributes:
        imageDimensions (int): The height and width of the input images. Assumes square images
        
        
    Methods:
        forward(inputs): Defines the forward pass of the neural network
    """
    def __init__(self, imageDimensions:int=28, **kwargs):
        """
        Initializes the network architecture
        
        Args:
            imageDimensions (int): The dimensions (height and width) of the input images
        """
        super(ConvolutionalNeuralNetwork, self).__init__(**kwargs)
        self.imageDimensions = imageDimensions
        self.conv1 = Conv2d(1, 32, 3)
        self.conv2 = Conv2d(32, 64, 3)
        self.conv3 = Conv2d(64, 128, 3)
        self.conv4 = Conv2d(128, 128, 1)
        self.pool  = MaxPool2d(2, stride=2)
        self.fc1   = Linear(128, 64)
        self.fc2   = Linear(64, 32)
        self.fc3   = Linear(32, 10)
        self.activation = ReLU()
        self.softmax    = softmax
        
    def forward(self, inputs):
        """
        Defines the forward pass of the neural network

        Args:
            inputs (torch.Tensor): The input data

        Returns:
            outputs (torch.Tensor): The output of the network after applying the softmax function
        """
        x = self.pool(self.activation(self.conv1(inputs)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.activation(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        outputs = self.softmax(x, dim=1)
        return outputs