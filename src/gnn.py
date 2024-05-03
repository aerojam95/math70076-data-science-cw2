#=============================================================================
# Programme: 
# Class for Graph neural network model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.nn import Linear, ReLU, Sequential, Tanh
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

class GraphNeuralNetwork(NeuralNetwork):
    """
    A feedforward Graph neural network for image classification

    Attributes:
        imageDimensions (int): The height and width of the input images. Assumes square images
        numClasses (int): Number of classes contained in the dataset that the model will be used on
        predEdge (bool): To calculate edges of the graph representation of the input images
        
    Methods:
        forward(inputs): Defines the forward pass of the neural network
        precomputeAdjacencyImages(imageDimensions, numClasses): 
    """
    def __init__(self, imageDimensions:int=28, numClasses:int=10, predEdge:bool=False, **kwargs):
        """
        Initializes the Graph neural network architecture
        
        Args:
            imageDimensions (int): The dimensions (height and width) of the input images
            numClasses (int): Number of classes contained in the dataset that the model will be used on
            predEdge (bool): To calculate edges of the graph representation of the input imagespredEdge
        """
        super(GraphNeuralNetwork, self).__init__(**kwargs)
        self.predEdge = predEdge
        N = imageDimensions ** 2 # Number of pixels in the image
        self.fc = Linear(N, numClasses, bias = False)
        # Create the adjacency matrix of size (N X N)
        if self.predEdge is True:
            # Learn the adjacency matrix (learn to predict the edge between any pair of pixels)
            col, row = np.meshgrid(np.arange(imageDimensions), np.arange(imageDimensions))
            coord = np.stack((col, row), axis = 2).reshape(-1, 2)  # (N X N x 2)
            coordNormalised = (coord - np.mean(coord, axis = 0)) / (np.std(coord, axis = 0) + 1e-5) # Normalise
            coordNormalised = torch.from_numpy(coordNormalised).float() # (N X N x 2)
            adjacencyMatrix = torch.cat((coordNormalised.unsqueeze(0).repeat(N, 1,  1), coordNormalised.unsqueeze(1).repeat(1, N, 1)), dim=2) # (N X N x N X N x 4)
            self.predEdgeFc = Sequential(Linear(4, 64),ReLU(), Linear(64, 1), Tanh())
            self.register_buffer("adjacencyMatrix", adjacencyMatrix) # Not model paramater during training
        else:
            # Use a pre-computed adjacency matrix
            A = self.precomputeAdjacencyImages(imageDimensions,numClasses)
            self.register_buffer("A", A) # Not model paramater during training
        self.softmax = softmax

    def forward(self, inputs):
        """
        Defines the forward pass of the neural network

        Args:
            inputs (torch.Tensor): The input data  (batchSize x 1 x imageDimensions x imageDimensions)

        Returns:
            outputs (torch.Tensor): The output of the network after applying the softmax function
        """
        B = inputs.size(0)
        if self.predEdge:
            self.A = self.predEdgeFc(self.adjacencyMatrix).squeeze() # (N X N x N X N) --> predicted edge map
        avgNeighborFeatures = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), 
                                            inputs.view(B, -1, 1)).view(B, -1)) # (64 X N X N)
        logits = self.fc(avgNeighborFeatures)
        outputs = self.softmax(logits, dim=1)
        return outputs

    @staticmethod
    # Static method knows nothing about the class and just deals with the parameters.
    def precomputeAdjacencyImages(imageDimensions:int=28, numClasses:int=10):
        """
        Initializes the Graph neural network architecture
        
        Args:
            imageDimensions (int): The dimensions (height and width) of the input images
            numClasses (int): Number of classes contained in the dataset that the model will be used on
            
        Returns:
            AHat (torch.Tensor):Adjacency matrix
        """
        col, row = np.meshgrid(np.arange(imageDimensions), np.arange(imageDimensions)) # (N X N)
        coord = np.stack((col, row), axis = 2).reshape(-1, 2) / imageDimensions # (N X N x 2) --> normalise
        dist = cdist(coord, coord) # compute distance between every pair of pixels
        sigma = 0.05 * np.pi # width of the Gaussian (can be a hyperparameter while training a model)
        A = np.exp(-dist / sigma ** 2) # adjacency matrix of spatial similarity
        A[A < 0.01] = 0 # suppress values less than 0.01
        A = torch.from_numpy(A).float()
        # Normalisation
        D = A.sum(1)  # nodes degree (N,)
        DHat = (D + 1e-5) ** (-0.5)
        AHat = DHat.view(-1, 1) * A * DHat.view(1, -1)  # N,N
        AHat[AHat > 0.0001] = AHat[AHat > 0.0001] - 0.2
        return AHat