#=============================================================================
# Programme: 
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# Custom modules
from logger import logProgress
from data_loader import loadImages, loadLabels
from gnb import GaussianNaiveBayes
from knn  import kNearestNeighbours

#=============================================================================
# Functions
#=============================================================================
    

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
metadataFilePath = "configurations.json"

# Pixel normalisation value
pixels = 255.

# Seed for repeatable random initialisations
seed = np.random.seed(123456789)

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
    with open(metadataFilePath, 'w') as metadata_file:
        json.dump(jsonData, metadata_file, indent=4)
    logProgress("Updated metadata file")
    
    # Paths to the downloaded data files
    trainImagesPath = os.path.join(dataPath, "train-images-idx3-ubyte.gz")
    trainLabelsPath = os.path.join(dataPath, "train-labels-idx1-ubyte.gz")
    testImagesPath  = os.path.join(dataPath, "t10k-images-idx3-ubyte.gz")
    testLabelsPath  = os.path.join(dataPath, "t10k-labels-idx1-ubyte.gz")

    # Load the datasets
    logProgress("Loading data...")
    trainImages = loadImages(trainImagesPath)
    trainLabels = loadLabels(trainLabelsPath)
    testImages  = loadImages(testImagesPath)
    testLabels  = loadLabels(testLabelsPath)
    logProgress("Loaded data")

    
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
    logProgress("Running Naive-Bayes...")
    GNB = GaussianNaiveBayes()
    GNB.run(trainImages, trainLabels, testImages, testLabels)
    logProgress("Naive-Bayes completed")
    
    GNB.saveModel(f"{outputModelPath}{runNumber}_naive_bayes_model_parameters_.txt")
    logProgress("Naive-Bayes model saved")

    GNB.saveValidation(f"{outputValPath}{runNumber}_naive_bayes_validation_results.txt")
    logProgress("Naive-Bayes validation accuracy saved")
    
    
    # k-Nearest neighbours
    logProgress("Training k-Nearest neighbours...")
    kNN = kNearestNeighbours(k=1, nSplits=10)
    kNN.train(f"{outputFigPath}{runNumber}_", trainImages, trainLabels, kmin=1, kmax=20)
    logProgress("k-Nearest neighbours training completed")
    
    logProgress("Validating k-Nearest neighbours...")
    kNN.evaluate(testImages, testLabels)
    logProgress("k-Nearest neighbours validation completed")
    
    kNN.saveModel(f"{outputModelPath}{runNumber}_knn_model_parameters_.txt")
    logProgress("k-Nearest neighbours model saved")

    kNN.saveValidation(f"{outputValPath}{runNumber}_knn_validation_results.txt")
    logProgress("k-Nearest neighbours validation accuracy saved")
    
    # Neural Network
    
    # Convolutional neural network
    
    # Geometric neural network 
    
    
    #==========================================================================
    # Programme completion
    #==========================================================================

    logProgress("Completed programme")