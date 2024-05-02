#=============================================================================
# Programme: 
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
import json
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST   
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import torch.optim as optim

# Custom modules
from logger import logProgress
from data_loader import loadImages, loadLabels
from gnb import GaussianNaiveBayes
from knn  import kNearestNeighbours
from nn import NeuralNetwork

#=============================================================================
# Functions
#=============================================================================
    

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
metadataFilePath = "configurations.json"

# Size of image dimensons for Fashion MNIST dataset
imageDimensions=28

# Pixel normalisation value
pixels = 255.

# Seed for repeatable random initialisations
seed = np.random.seed(123456789)
torch.manual_seed(123456789)

# Set loss function for DL methods
criterion = CrossEntropyLoss()

#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Start programme
    #==========================================================================
    
    # Logging
    logProgress("Starting programme...")
    
    #==========================================================================
    # Data loading
    #==========================================================================

    # Reading the JSON configuration file
    logProgress("Importing metadata...")
    with open(metadataFilePath, "r") as metadata_file:
        jsonData = json.load(metadata_file)
    logProgress("Imported metadata")
    
    # Setting metadata variables
    loggingName     = jsonData["loggingName"]
    runNumber       = jsonData["runNumber"]
    dataPath        = jsonData["dataPath"]
    outputFigPath   = jsonData["outputFigPath"]
    outputValPath   = jsonData["outputValPath"]
    outputModelPath = jsonData["outputModelPath"]
    
    # Update metadata file for next run
    logProgress("Updating metadata file...")
    jsonData["runNumber"] = str(int(runNumber) + 1)
    with open(metadataFilePath, "w") as metadata_file:
        json.dump(jsonData, metadata_file, indent=4)
    logProgress("Updated metadata file")
    
    ## Load data for ML methods
    
    # Paths to the downloaded data files
    trainImagesPath = os.path.join(dataPath, "train-images-idx3-ubyte.gz")
    trainLabelsPath = os.path.join(dataPath, "train-labels-idx1-ubyte.gz")
    testImagesPath  = os.path.join(dataPath, "t10k-images-idx3-ubyte.gz")
    testLabelsPath  = os.path.join(dataPath, "t10k-labels-idx1-ubyte.gz")

    # # Load the datasets
    logProgress("Loading ML data...")
    trainImages = loadImages(trainImagesPath)
    trainLabels = loadLabels(trainLabelsPath)
    testImages  = loadImages(testImagesPath)
    testLabels  = loadLabels(testLabelsPath)
    logProgress("Loaded ML data")
    
    ## Load data for DL methods
    
    # Transformations applied on each image => 
    # first make them a tensor, then normalize them in the range -1 to 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset and split it into training and validation parts
    trainDataFull  = FashionMNIST(root=dataPath, train=True, transform=transform, download=True)
    trainData, valData = data.random_split(trainDataFull , [50000, 10000])

    # Loading the test set
    testData = FashionMNIST(root=dataPath,train=False, transform=transform, download=True)

    # Define a set of data loaders 
    trainData  = DataLoader(trainData, batch_size=64, shuffle=True, drop_last=False)
    valData    = DataLoader(valData, batch_size=64, shuffle=False, drop_last=False)
    testData   = DataLoader(testData, batch_size=64, shuffle=False, drop_last=False)
    
    #==========================================================================
    # Data pre-processing
    #==========================================================================
    
    # pre-process images
    logProgress("Pre-processing image data...")
    trainImages = trainImages.reshape(trainImages.shape[0], -1) / pixels
    testImages = testImages.reshape(testImages.shape[0], -1) / pixels
    logProgress("Pre-processed image data")
    
    #==========================================================================
    # Methods to run
    #==========================================================================
    
    # Gaussian Naive-Bayes
    logProgress("Running Gaussian Naive-Bayes...")
    GNB = GaussianNaiveBayes(smoothing=1000.0)
    GNB.train(trainImages, trainLabels)
    logProgress("Naive-Bayes completed")
    
    logProgress("Validating Gaussian Naive-Bayes...")
    accuracyGNB = GNB.evaluate(testImages, testLabels)
    logProgress("Gaussian Naive-Bayes validation completed")
    
    GNB.saveModel(f"{outputModelPath}{runNumber}_gaussian_naive_bayes_model_parameters_.txt")
    logProgress("Gaussian Naive-Bayes model saved")
    
    
    # k-Nearest neighbours
    logProgress("Training k-Nearest neighbours...")
    kNN = kNearestNeighbours(k=1, nSplits=2)
    kNN.train(f"{outputFigPath}{runNumber}_", trainImages, trainLabels, kmin=1, kmax=3)
    logProgress("k-Nearest neighbours training completed")
    
    logProgress("Validating k-Nearest neighbours...")
    accuracykNN = kNN.evaluate(testImages, testLabels)
    logProgress("k-Nearest neighbours validation completed")
    
    kNN.saveModel(f"{outputModelPath}{runNumber}_knn_model_parameters_.txt")
    logProgress("k-Nearest neighbours model saved")
    
    
    # Neural Network
    logProgress("Training neural network...")
    NN = NeuralNetwork(imageDimensions=imageDimensions)
    optimizer = optim.Adam(NN.parameters(), lr=0.001)
    NN.trainModel(trainData, valData, criterion, optimizer, f"{outputFigPath}{runNumber}_", epochs=2)
    logProgress("neural network training completed")

    logProgress("Validating neural network...")
    accuracNN = NN.evaluate(testData)
    logProgress("Neural network validation completed")

    NN.saveModel(f"{outputModelPath}{runNumber}_nn_model.pth")
    logProgress("Neural network model saved")
    
    
    # Convolutional neural network
    
    
    
    # Geometric neural network 
    
    
    #==========================================================================
    # Programme completion
    #==========================================================================

    logProgress("Completed programme")