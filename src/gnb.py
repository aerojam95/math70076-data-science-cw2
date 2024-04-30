#=============================================================================
# Programme: 
# Class for Gaussian Naive-Bayes model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
import sklearn.metrics 
import sklearn.model_selection
import sklearn.preprocessing

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
    A Gaussian Naive Bayes classifier to perform classification tasks on image data.

    Attributes:
        smoothing (float): A smoothing factor to avoid division by zero errors in variance calculations.
    
    Methods:
        run(trainImages, trainLabels, valImages, valLabels): Trains the classifier and predicts the class labels for validation data.
        saveModel(filename): Saves the model parameters to a file.
        saveValidation(filename): Saves validation results to a file.
    """
    def __init__(self, smoothing:float=1000., **kwargs):
        """
        Initializes the GaussianNaiveBayes classifier with optional smoothing.

        Args:
            smoothing (float): The variance smoothing factor; defaults to 1000.0.
        """
        self.smoothing   = smoothing
        self.m           = []
        self.s           = []
        self.count       = []
        self.prior       = []
        self.count       = []
        self.pred        = []
        self.likelihood  = []
        self.lcs         = []
        
    def run(self, trainImages, trainLabels, valImages, valLabels):
        """
        Trains the Gaussian Naive Bayes model using the provided training data and then uses the trained model to predict the validation data.

        Args:
            trainImages (array-like): Array of training images.
            trainLabels (array-like): Array of labels corresponding to the training images.
            valImages (array-like): Array of validation images.
            valLabels (array-like): Array of labels corresponding to the validation images.

        Returns:
            None: Outputs are stored in instance variables.
        """
        # Initialise class data attributes
        self.classes     = np.unique(trainLabels)
        self.trainImages = trainImages
        self.valImages  = valImages
        self.trainLabels = trainLabels
        self.valLabels  = valLabels
        
        # Training data for model parameteres
        for category in self.classes:
            sep = self.trainLabels == category
            self.count.append(np.sum(sep))
            self.prior.append(np.mean(sep))
            self.m.append(np.mean(self.trainImages[sep], axis=0))
            self.s.append(np.std(self.trainImages[sep], axis=0))
                    
        # Prediction on the valiation data
        for n in range(len(self.valLabels)):
            classifier = []
            sample = self.valImages[n]
            ll = []
            for index, category in enumerate(self.classes):
                m1 = self.m[index]
                var = np.square(self.s[index]) + self.smoothing
                prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - m1)/(2 * var))
                result = np.sum(np.log(prob))
                classifier.append(result)
                ll.append(prob)
            self.pred.append(np.argmax(classifier))
            self.likelihood.append(ll)
            self.lcs.append(classifier)
        
        # Model metrics
        self.cm = sklearn.metrics.confusion_matrix(valLabels, self.pred)
        self.accuracy = round((sum(np.diagonal(self.cm)) / len(self.pred)), 4)
        
        return None
    
    def saveModel(self, filename:str='naive_bayes_model_parameters.txt'):
        """
        Saves the model's parameters to a text file.

        Args:
            filename (str): The filename or path to save the model parameters to; defaults to 'naive_bayes_model_parameters.txt'.
        
        Returns:
            None
        """
        with open(filename, 'w') as file:
            file.write(f'Smoothing parameter: {self.smoothing}\n')
            for index, cls in enumerate(self.classes):
                file.write(f'Class {cls}:\n')
                file.write(f'Prior: {self.prior[index]}\n')
                file.write('Mean: [' + ', '.join(f'{m:.2f}' for m in self.m[index]) + ']\n')
                file.write('Std Deviation: [' + ', '.join(f'{s:.2f}' for s in self.s[index]) + ']\n\n')
        return None

    def saveValidation(self, filename:str='naive_bayes_validation_results.txt'):
        """
        Saves the validation results to a text file.

        Args:
            filename (str): The filename or path to save the validation results to; defaults to 'naive_bayes_validation_results.txt'.

        Returns:
            None
        """
        with open(filename, 'w') as file:
            file.write(f'Accuracy: {self.accuracy * 100:.2f}%\n')
        return None