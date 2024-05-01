#=============================================================================
# Programme: 
# Class for k-Nearest neighbours model
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import numpy as np
import pandas as pd
import sklearn.metrics 
import sklearn.model_selection
import sklearn.preprocessing
import scipy.stats
import matplotlib.pyplot as plt
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

class kNearestNeighbours():
    
    def __init__(self, k:int=1, **kwargs):
        self.k               = k
        self.predictAccuracy = None
        
    def _getTrainingCurve(sef, CurvePath:str, trainingFrame, kmin:int=1, kmax:int=100):
        plt.subplots(figsize=(20, 10))
        plt.plot(np.mean(trainingFrame, axis = 0))
        plt.xticks(range(kmin - 1,kmax - 1), range(kmin, kmax))
        plt.grid(True)
        plt.title('Loss vs K value')
        plt.savefig(f'{CurvePath}knn.png')
        return None
    
    def _getKOpt(df):
        # Compute the sum of errors for each k value
        errors = df.sum(axis=0)
        # Find the index (k value) with the minimum sum of errors
        kOpt = errors.idxmin()
        return kOpt
    
    def train(self, CurvePath:str, trainImages, trainLabel, kmin:int=1, kmax:int=100):
        klist = range(kmin, kmax + 1)
        df = pd.DataFrame(index=range(len(trainLabel)), columns=range(len(klist)))
        for p in tqdm(range(len(trainLabel)), desc="Training Progress"):
            te = trainImages[p]
            te_lb = trainLabel[p]
            tr = np.delete(trainImages, p, 0)
            train_label = np.delete(trainLabel, p)
            diff = (tr - te)
            # Eculidean distance
            dis = np.einsum('ij, ij->i', diff, diff) 
            for i, k in enumerate(klist):
                near = train_label[np.argsort(dis)[:k]]
                pick = scipy.stats.mode(near)[0]
                if pick == te_lb:
                    df.iloc[p][i] = 0
                else:
                    df.iloc[p][i] = 1
        self._gettrainingCurve(df, kmin, kmax, CurvePath)
        self.k = self._getKOpt(df)
        return df
        
    def predict(self, trainImages, trainLabel, valImages, valLables, k:int=1):
        pred = []
        for w in range(len(valImages)):
            test_1 = valImages[w]
            diff = (trainImages - test_1)
            # Eculidean distance
            dist = np.einsum('ij, ij->i', diff, diff) 
            nearest_lbs = trainLabel[np.argsort(dist)[:k]]
            major = scipy.stats.mode(nearest_lbs)[0]
            pred.append(major)
        self.predictAccuracy = sklearn.metrics.accuracy_score(valLables, pred)
        return None
        
    def saveModel(self, filename:str='knn_model_parameters.txt'):
        """
        Saves the model's parameters to a text file.

        Args:
            filename (str): The filename or path to save the model parameters to; defaults to 'knn_model_parameters.txt'.
        
        Returns:
            None
        """
        with open(filename, 'w') as file:
            file.write(f'k: {self.k}\n')
        return None

    def saveValidation(self, filename:str='knn_validation_results.txt'):
        """
        Saves the validation results to a text file.

        Args:
            filename (str): The filename or path to save the validation results to; defaults to 'knn_validation_results.txt'.

        Returns:
            None
        """
        with open(filename, 'w') as file:
            file.write(f'Accuracy: {self.predictAccuracy * 100:.2f}%\n')
        return None