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
    
    GNB.saveModel(f"{outputModelPath}naive_bayes_model_parameters_{runNumber}.txt")
    logProgress("Naive-Bayes model saved")

    GNB.saveValidation(f"{outputValPath}naive_bayes_validation_results_{runNumber}.txt")
    logProgress("Naive-Bayes validation accuracy saved")
    
    
    # k-Nearest neighbours
    
    # Neural Network
    
    # Convolutional neural network
    
    # Geometric neural network 
    
    
    #==========================================================================
    # Programme completion
    #==========================================================================

    logProgress("Completed programme")