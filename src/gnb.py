#=============================================================================
# Programme: 
# Class for Gaussian Naive-Bayes model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
from  sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os

# Custom modules

#=============================================================================
# Functions
#=============================================================================
    
#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Classes
#=============================================================================

class GaussianNaiveBayes():
    """
    A Gaussian Naive Bayes classifier to perform classification tasks on image data

    Attributes:
        smoothing (float): A smoothing factor to avoid division by zero errors in variance calculations
    
    Methods:
        train(trainImages, trainLabels): Trains the classifier on test data
        evaluate(testImages, testLabels): evalutes trained model on test data
        predict(testImages): makes predictions on image inputs and classifies image based on model classes in trianing data
        saveModel(filename): Saves the model parameters to a file
        loadModel(filename): loads model parameters from an exists model .txt file
    """
    def __init__(self, smoothing:float=1000.):
        """
        Initializes the GaussianNaiveBayes classifier with optional smoothing

        Args:
            smoothing (float): The variance smoothing factor; defaults to 1000.0
        """
        self.classes           = None
        self.smoothing         = smoothing
        self.mean              = []
        self.standardDeviation = []
        self.count             = []
        self.prior             = []
        
    def train(self, trainImages, trainLabels):
        """
        Trains the Gaussian Naive Bayes model using the provided training data and then uses the trained model to predict the validation data

        Args:
            trainImages (array-like): Array of training images
            trainLabels (array-like): Array of labels corresponding to the training images

        Returns:
            None
        """
        # Initialise attributes
        self.classes           = np.unique(trainLabels)
        self.mean              = []
        self.standardDeviation = []
        self.count             = []
        self.prior             = []
        
        # Training data for model parameteres
        for category in tqdm(self.classes, desc="Training Progress"):
            sep = trainLabels == category
            self.count.append(np.sum(sep))
            self.prior.append(np.mean(sep))
            self.mean.append(np.mean(trainImages[sep], axis=0))
            self.standardDeviation.append(np.std(trainImages[sep], axis=0))
        return None
    
    def evaluate(self, testImages, testLabels):
        """
        Evaluates the Gaussian Naive Bayes model using the trained model on test data

        Args:
            testImages (array-like): Array of validation images
            testLabels (array-like): Array of labels corresponding to the validation images

        Returns:
            accuracy (float): test accuracy of the model using trained parameters
        """
        if self.classes is None:
            raise ValueError("Model has not been trained")
        else:
            pred = []
            likelihood = []
            lcs = []
            for n in tqdm(range(len(testLabels))):
                classifier = []
                sample = testImages[n]
                ll = []
                for index, _ in enumerate(self.classes):
                    m1 = self.mean[index]
                    var = np.square(self.standardDeviation[index]) + self.smoothing
                    prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - m1)/(2 * var))
                    result = np.sum(np.log(prob))
                    classifier.append(result)
                    ll.append(prob)
                pred.append(np.argmax(classifier))
                likelihood.append(ll)
                lcs.append(classifier)
            
            # Model metrics 
            cm = confusion_matrix(testLabels, pred)
            accuracy = round((sum(np.diagonal(cm)) / len(pred)), 4)
            print(f"Test accuracy for Gaussian Naive Bayes model: {accuracy * 100:04.2f}%")
            return accuracy
    
    def predict(self, testImages):
        """
        Predicts classification using the trained Gaussian Naive Bayes model

        Args:
            testImages (array-like): Array of images to predic classifications

        Returns:
            predictions (list): list of classifications for each argument image
        """
        if self.classes is None:
            raise ValueError("Model has not been trained")
        else:
            predications = []
            for n in tqdm(range(np.shape(testImages)[0])):
                classifier = []
                sample = testImages[n]
                ll = []
                for index, _ in enumerate(self.classes):
                    m1 = self.mean[index]
                    var = np.square(self.standardDeviation[index]) + self.smoothing
                    prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - m1)/(2 * var))
                    result = np.sum(np.log(prob))
                    classifier.append(result)
                    ll.append(prob)
                predications.append(np.argmax(classifier))
            return predications
    
    def saveModel(self, filename:str="naive_bayes_model_parameters.txt"):
        """
        Saves the model's parameters to a text file

        Args:
            filename (str): The filename or path to save the model parameters to; defaults to "naive_bayes_model_parameters.txt"
        
        Returns:
            None
        """
        if self.classes is None:
            raise ValueError("Model has not been trained")
        else:
            with open(filename, "w") as file:
                file.write(f"Smoothing parameter: {self.smoothing}\n")
                for index, cls in enumerate(self.classes):
                    file.write(f"Class {cls}:\n")
                    file.write(f"Prior: {self.prior[index]}\n")
                    file.write("Mean: [" + ", ".join(f"{m:.2f}" for m in self.mean[index]) + "]\n")
                    file.write("Std Deviation: [" + ", ".join(f"{s:.2f}" for s in self.standardDeviation[index]) + "]\n\n")
            return None

    def loadModel(self, filename:str="naive_bayes_model_parameters.txt"):
        """
        Loads the model from the specified path

        Args:
            filename (str): The path from where to load the model
            
        Returns:
            None
        """
        if os.path.isfile(filename) is False:
            raise ValueError("Model file does not exist")
        else:
            with open(filename, "r") as file:
                lines = file.readlines()
            
            self.smoothing = float(lines[0].strip().split(": ")[1])
            self.classes = []
            self.prior = []
            self.mean = []
            self.standardDeviation = []

            i = 1
            while i < len(lines):
                if "Class" in lines[i]:
                    cls = lines[i].strip().split(" ")[1].strip(":")
                    self.classes.append(int(cls))
                    i += 1
                    self.prior.append(float(lines[i].strip().split(": ")[1]))
                    i += 1
                    means = lines[i].strip().split(": ")[1]
                    self.mean.append([float(mean) for mean in means.strip("[]").split(", ")])
                    i += 1
                    stds = lines[i].strip().split(": ")[1]
                    self.standardDeviation.append([float(std) for std in stds.strip("[]").split(", ")])
                    i += 2
                else:
                    i += 1
            self.classes = np.unique(self.classes)
            
            return None