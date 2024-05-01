#=============================================================================
# Programme: 
# Class for k-Nearest neighbours model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

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

class kNearestNeighbours:
    """
    A class for K-Nearest Neighbors classification using Scikit-Learn
    Implemented using scikit-learn for computational efficiency

    Attributes:
        k (int): The number of neighbors to consider for the k-NN algorithm.
        model (KNeighborsClassifier): The Scikit-Learn k-NN classifier.
        predictAccuracy (float): The accuracy of the model on the validation set, set after evaluation.

    Methods:
        train(CurvePath, trainImages, trainLabels, kmin, kmax): Trains the k-NN model over a range of k values.
        predict(testImages): Predicts labels for the given test images.
        evaluate(testImages, testLabels): Evaluates the model on the test data.
        saveModel(filename): Saves the optimal k value to a file.
        saveValidation(filename): Saves the model's accuracy to a file.
    """

    def __init__(self, k:int=1, nSplits:int=5):
        """
        Initializes the kNearestNeighbours class with a specified number of neighbors.

        Args:
            k (int): The number of neighbors to use for the classifier.
            nSplits (int): The number of folds for cross-validation.
        """
        self.k = k
        self.nSplits = nSplits
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.predictAccuracy = None

    def _getTrainingCurve(self, accuracies, CurvePath:str, kmin:int=1, kmax:int=100):
        """
        Generates and saves a plot of training accuracies versus different values of k.

        Args:
            accuracies (list): A list of accuracy scores corresponding to different k values.
            CurvePath (str): The path where the plot image will be saved.
            kmin (int): The minimum k value (inclusive).
            kmax (int): The maximum k value (inclusive).
        """
        plt.subplots(figsize=(20, 10))
        plt.plot(range(kmin, kmax + 1), accuracies)
        plt.xticks(range(kmin, kmax + 1))
        plt.xlabel("k")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.title('Loss vs k')
        plt.savefig(f'{CurvePath}knn_training_curve.png')
        plt.close()

    def train(self, CurvePath:str, trainImages, trainLabels, kmin:int=1, kmax:int=100):
        """
        Trains the k-NN model using a range of k values to find the optimal k. Generates a training curve using cross-validation.

        Args:
            CurvePath (str): The directory path to save the training curve plot.
            trainImages (np.array): The training images dataset.
            trainLabels (np.array): The labels corresponding to the training images.
            kmin (int): The minimum k value to consider.
            kmax (int): The maximum k value to consider.
            n_splits (int): The number of folds for cross-validation.

        Returns:
            list: A list of average cross-validation scores for each k value from kmin to kmax.
        """
        accuracies = []
        losses = []
        kOpt = kmin
        bestAccuracy = 0

        for k in tqdm(range(kmin, kmax + 1), desc="Training Progress"):
            model = KNeighborsClassifier(n_neighbors=k)
            # Perform cross-validation and compute the mean accuracy
            cvScores = cross_val_score(model, trainImages, trainLabels, cv=self.nSplits, scoring='accuracy')
            meanAccuracy = np.mean(cvScores)
            accuracies.append(meanAccuracy)
            losses.append(1 - meanAccuracy)

            # Update best k based on highest mean accuracy
            if meanAccuracy > bestAccuracy:
                bestAccuracy = meanAccuracy
                kOpt = k

        # Update model with the optimal k
        self.k = kOpt
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(trainImages, trainLabels)

        # Plot the training curve
        self._getTrainingCurve(losses, CurvePath, kmin, kmax)

        return accuracies

    def predict(self, testImages):
        """
        Predicts the labels for the given test images using the trained k-NN model.

        Args:
            testImages (np.array): The test images dataset.

        Returns:
            np.array: The predicted labels for the test images.
        """
        if self.model:
            return self.model.predict(testImages)
        else:
            raise ValueError("Model has not been trained yet!")

    def evaluate(self, testImages, testLabels):
        """
        Evaluates the model on the test dataset using the accuracy metric.

        Args:
            testImages (np.array): The test images dataset.
            testLabels (np.array): The true labels for the test images.
        
        Returns:
            float: The accuracy of the model on the test dataset.
        """
        predictions = self.predict(testImages)
        self.predictAccuracy = accuracy_score(testLabels, predictions)
        return self.predictAccuracy

    def saveModel(self, filename:str='knn_model_parameters.txt'):
        """
        Saves the model's optimal k value to a specified file.

        Args:
            filename (str): The filename or path where the model parameters should be saved.
        """
        with open(filename, 'w') as file:
            file.write(f'k: {self.k}\n')
            file.write(f'nSplits: {self.nSplits}\n')

    def saveValidation(self, filename:str='knn_validation_results.txt'):
        """
        Saves the validation results to a specified file.

        Args:
            filename (str): The filename or path where the validation results should be saved.
        """
        if self.predictAccuracy is not None:
            with open(filename, 'w') as file:
                file.write(f'Accuracy: {self.predictAccuracy * 100:.2f}%\n')
        else:
            raise ValueError("Model has not been evaluated yet!")
