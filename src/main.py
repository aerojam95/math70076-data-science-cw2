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
from knn import kNearestNeighbours
from nn  import NeuralNetwork
from cnn import ConvolutionalNeuralNetwork
from gnn import GraphNeuralNetwork
from predict_plotter import predictPlotter

#=============================================================================
# Functions
#=============================================================================
    

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
metadataFilePath = "configurations.json"

# Use GPU resources for DL model if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Size of image dimensons for Fashion MNIST dataset
imageDimensions=28

# Pixel normalisation value
pixels = 255.

# Seed for repeatable random initialisations
seed = np.random.seed(123456789)
torch.manual_seed(123456789)

# Set loss function for DL methods
criterion = CrossEntropyLoss()

# Class dictionary
classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Number of classes in Fashion MNIST dataset
numClasses= 10

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
    print(trainImages.shape)
    print(trainLabels.shape)
    
    # Take random samples here for later ML predictions
    randomIndices = [0, 1000]
    randomImages =  [testImages[index] for index in randomIndices]
    randomLabels =  [testLabels[index] for index in randomIndices]
    
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
    
    GNB.saveModel(f"{outputModelPath}{loggingName}_{runNumber}_gaussian_naive_bayes_model_parameters.txt")
    logProgress("Gaussian Naive-Bayes model saved")
    
    predictionGNB = GNB.predict(np.expand_dims(randomImages[0], axis=0))
    predictPlotter(randomImages[0].reshape(28, 28), randomLabels[0], predictionGNB[0], classes, f"{outputValPath}{loggingName}_{runNumber}_prediction_gnb.png")
    logProgress("Gaussian Naive-Bayes example prediction")
    
    
    # k-Nearest neighbours
    logProgress("Training k-Nearest neighbours...")
    kNN = kNearestNeighbours(k=1, nSplits=10)
    kNN.train(f"{outputFigPath}{loggingName}_{runNumber}_", trainImages, trainLabels, kmin=1, kmax=20)
    logProgress("k-Nearest neighbours training completed")
    
    logProgress("Validating k-Nearest neighbours...")
    accuracykNN = kNN.evaluate(testImages, testLabels)
    logProgress("k-Nearest neighbours validation completed")
    
    kNN.saveModel(f"{outputModelPath}{loggingName}_{runNumber}_knn_model_parameters.txt")
    logProgress("k-Nearest neighbours model saved")
    
    predictionkNN = kNN.predict(np.expand_dims(randomImages[1], axis=0))
    predictPlotter(randomImages[1].reshape(28, 28), randomLabels[1], predictionkNN[0], classes, f"{outputValPath}{loggingName}_{runNumber}_prediction_knn.png")
    logProgress("k-Nearest neighbours example prediction")
    
    
    # Neural Network
    logProgress("Training neural network...")
    NN = NeuralNetwork(imageDimensions=imageDimensions, numClasses=numClasses)
    optimizer = optim.Adam(NN.parameters(), lr=0.001)
    NN.trainModel(trainData, valData, criterion, optimizer, f"{outputFigPath}{loggingName}_{runNumber}_", epochs=50)
    logProgress("neural network training completed")

    logProgress("Validating neural network...")
    accuracyNN = NN.evaluate(testData)
    logProgress("Neural network validation completed")

    NN.saveModel(f"{outputModelPath}{loggingName}_{runNumber}_nn_model.pth")
    logProgress("Neural network model saved")
    
    predictionNN = NN.predict(testData.dataset[2000][0])
    predictPlotter(testData.dataset[2000][0].squeeze(), testData.dataset[2000][1], predictionNN, classes, f"{outputValPath}{loggingName}_{runNumber}_prediction_NN.png")
    logProgress("Neural network example prediction")
    
    
    # Convolutional neural network
    logProgress("Training convolutional neural network...")
    CNN = ConvolutionalNeuralNetwork()
    optimizer = optim.Adam(CNN.parameters(), lr=0.001)
    CNN.trainModel(trainData, valData, criterion, optimizer, f"{outputFigPath}{loggingName}_{runNumber}_", epochs=50)
    logProgress("Convolutional neural network training completed")

    logProgress("Validating convolutional neural network...")
    accuracyCNN = CNN.evaluate(testData)
    logProgress("Convolutional neural network validation completed")

    CNN.saveModel(f"{outputModelPath}{loggingName}_{runNumber}_cnn_model.pth")
    logProgress("Convolutional neural network model saved")
    
    predictionCNN = CNN.predict(testData.dataset[3000][0].unsqueeze(1))
    predictPlotter(testData.dataset[3000][0].squeeze(), testData.dataset[3000][1], predictionCNN, classes, f"{outputValPath}{loggingName}_{runNumber}_prediction_CNN.png")
    logProgress("Convolutional neural network example prediction")
    
    
    # Graph neural network 
    logProgress("Training Graph neural network...")
    GNN = GraphNeuralNetwork(imageDimensions=imageDimensions, numClasses=numClasses, predEdge=True)
    optimizer = optim.Adam(GNN.parameters(), lr=0.001)
    GNN.trainModel(trainData, valData, criterion, optimizer, f"{outputFigPath}{loggingName}_{runNumber}_", epochs=10)
    logProgress("Graph neural network training completed")

    logProgress("Validating geometric neural network...")
    accuracyGNN = GNN.evaluate(testData)
    logProgress("Graph neural network validation completed")

    GNN.saveModel(f"{outputModelPath}{loggingName}_{runNumber}_gnn_model.pth")
    logProgress("Graph neural network model saved")
    
    predictionGNN = GNN.predict(testData.dataset[4000][0])
    predictPlotter(testData.dataset[4000][0].squeeze(), testData.dataset[4000][1], predictionGNN, classes, f"{outputValPath}{loggingName}_{runNumber}_prediction_GNN.png")
    logProgress("Graph neural network example prediction")
    
    #==========================================================================
    # Programme completion
    #==========================================================================

    logProgress("Completed programme")