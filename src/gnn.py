#=============================================================================
# Programme: 
# Class for Graph neural network model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch_geometric.nn import GATConv


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
    A feedforward neural network for image classification

    Attributes:
        imageDimensions (int): The height and width of the input images. Assumes square images
        
        
    Methods:
        forward(inputs): Defines the forward pass of the neural network
    """
    def __init__(self, imageDimensions:int=28, inChannels: int = 1, hiddenDim: int = 152, numClasses: int = 10, **kwargs):
        """
        Initializes the network architecture
        
        Args:
            imageDimensions (int): The dimensions (height and width) of the input images
        """
        super(GraphNeuralNetwork, self).__init__(**kwargs)
        self.imageDimensions = imageDimensions
        self.inChannels = inChannels
        self.hiddenDim  = hiddenDim
        self.numClasses = numClasses
        self.conv1 = GATConv(in_channels=inChannels, out_channels=hiddenDim)
        self.conv2 = GATConv(in_channels=hiddenDim, out_channels=hiddenDim)
        self.conv3 = GATConv(in_channels=inChannels + hiddenDim, out_channels=hiddenDim)
        self.fc = Sequential(
            Linear(inChannels + 3 * hiddenDim, 256),
            ReLU(True),
            Linear(256, 32),
            ReLU(True),
            Linear(32, 32),
            ReLU(True),
            Linear(32, numClasses),
            softmax(dim=1)
        )
        
    def forward_one_base(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        x0 = node_features
        x1 = self.conv1(x0, edge_indices)
        x2 = self.conv2(x1, edge_indices)
        x0_x2 = torch.cat((x0, x2), dim=-1)
        x3 = self.conv3(x0_x2, edge_indices)
        x0_x1_x2_x3 = torch.cat((x0, x1, x2, x3), dim=-1)
        return x0_x1_x2_x3

    def forward(self, batch_node_features: list[torch.Tensor], batch_edge_indices: list[torch.Tensor]) -> torch.Tensor:
        features_list = []
        for node_features, edge_indices in zip(batch_node_features, batch_edge_indices):
            features_list.append(self.forward_one_base(node_features=node_features, edge_indices=edge_indices))
        features = torch.stack(features_list, dim=0)  # BATCH_SIZE x NUM_NODES x NUM_FEATURES
        features = features.mean(dim=1)  # readout operation [BATCH_SIZE x NUM_FEATURES]
        logits = self.fc(features)
        return logits