# A comparative study of machine and deep learning models for image classification
This repository presents the second coursework for the MATH70076 Data Science module at Imperial College London, where the project showcases different machine and deep learning models for image classification where the models' performances and complexities are evaluated.


## Project summary

This data project looked at comparing the performance of different machine and deep learning models. The complexity of the models was also considered, i.e. how many parameters each of the models had for the task of image classification. Furthermore, each model was implemented as a Python3 class and the implemented models are used as modules within the main programme of the data project. 

To begin the models all used the same datasets for training and testing, respectively. The datasets used for this study were the [MNIST-Fashion](https://github.com/zalandoresearch/fashion-mnist) datasets. This dataset contained images of $10$ categories (classes) of clothing items. The models would take the images as inputs and then predict a classification of the image to one of the $10$ categories. Five different methods were investigated, which included a Gaussian naive-Bayes algorithm, a $k$-nearest neighbours algorithm, a neural network, a convolutional neural network, and a graph neural network. The classes were built to be reasonably general to be of use to end users. As such, the implemented classes all have class methods for training, evaluation, prediction, saving, and loading the model or model parameters. The two machine learning algorithms, Gaussian Naive-Bayes and $k$-Nearest neighbours, save and load model parameters; and the deep learning methods save and load the full models. The deep learning models, the neural network, convolutional neural network, and graph neural network, where all implemented in the PyTorch deep learning framework for Python3. These models were implemented for general image sizes, but the images would need to be black and white for these models to run.

The final part of the project was then to compare the methods and to provide a broad recommendation on which method would be best for image classification of black and white images for limited tuning. The results are presented below in the table and it is proposed that the neural network is used for image classification of black and white images given its performance. It is a more complex model than the $k$-nearest neighbours with a similar performance but the $k$-nearest neighbours algorithm will have a large computation time as it must compute the distances of between a sample and the other elements of the dataset to make the prediction. Thus, once training is complete the neural network is deemed to be the best classifier in this data project. The results in the table below are repoducible by running the ```main.py``` programme as it is in the repository.

| Model       | Accuracy (%)       | Number of model parameters       |
|----------------|----------------|----------------|
| Gaussian naive-Bayes  | 61.46  | 20  |
| $k$-nearest neighbours  | 85.77  | -  |
| Neural network  | 86.18  | 118,282  |
| Convolutional neural network  | 61.97  | 1,593,620  |
| Graph neural network  | 35.47  | 126,507  |


## Data
 
- The dataset used for this comparative study was the [MNIST-Fashion](https://github.com/zalandoresearch/fashion-mnist) dataset where it can be downloaded directly from the repository's [README](https://github.com/zalandoresearch/fashion-mnist) page
- The [raw](data/raw/) folder contains all the raw downloaded data from the dataset's [README](https://github.com/zalandoresearch/fashion-mnist) page, where it has been downloaded into this local directory for processing ready for use in the models that will be used in this study
- The test and train datasets for the machine learning models, Gaussian naive-Bayes and $k$-nearest neighbours, are loaded from the downloaded datasets in the [raw](data/raw/) folder and imported as numpy arrays
- The test and train datasets for the deep learning models, neural network, convolutional neural network, and graph neural network, are loaded from the PyTorch utils and imported as PyTorch datasets
- The data are loaded twice because the different families of models have different dataset inputs into their classes for processeing the inputted data 


## Code

The code to produce the models and results for the comparative study can be found in [src](src/).

### Python virtual environment

Before using the code it is best to setup and start a Python virtual environment in order to avoid potential package clashes using the [requirements](src/requirements.txt) file:

```
# Navigate into the data project directory

# Create a virtual environment
python3 -m venv <env-name>

# Activate virtual environment
source <env-name>/bin/activate

# Install dependencies for code
pip3 install -r requirements.txt

# When finished with virtual environment
deactivate
```

### Configuration

The amount of hardcoding has been reduced as much as possible by creating a [configuration](src/configurations.json). This means that if the location of the main programme outputs, the data files, or the naming attributes for outputs files need to change they are updated in this ```.json``` file. Here is an additional example of a configuration object which can be used in place of the [current object](src/configurations.json):

```
# configurationns .json object
{
    "loggingName": "example",
    "runNumber": "1",
    "dataPath": "../data/raw/",
    "outputFigPath": "../outputs/learning-curves/",
    "outputValPath": "../outputs/test-predictions/",
    "outputModelPath": "../outputs/saved-models/"
}
```

- ```loggingName```: the name assigned to the outputs of the main programme that wishes to be run for end user file identification
- ```runNumber```: the run number of this configuration file, set to prevent overwriting previous runs, this can be re-set to 1 for a new run campaign but is not strictly required 
- ```dataPath```: the relative path to the data files used to be loaded into the numpy arrays to used for the machine learning models
- ```outputFigPath```: the relative path for saving the training curves for the models
- ```outputValPath```: the relative path for saving the prediction image outputs the models
- ```outputModelPath```: the relative path for saving the models or model parameters for each of the models

### Hyperparameters

The following hyperparameters can be easily tuned and modified in the [```main.py```](src/main.py) programme for each of the models before executing the programme. Further hyperparameters and model choices can be changed but require the user to modify the models' class codes in the respective Python3 modules.

### Gaussian Naive-Bayes

- ```smoothing```: hyperparameter to prevent numerical instabilites when calculating the exponent of the Gaussian distribution used in the model (preventing division of zero when the variance is very small)

### $k$-Nearest neighbours

- ```kmin```: minimum $k$ neighbours to test when finding the optimimum number of neighbours for a given dataset
- ```kmax```: maximum $k$ neighbours to test when finding the optimimum number of neighbours for a given dataset
- ```nSplits```: number of cross-validation splits when testing each $k$ neighbours in finding the optimimum number of neighbours for a given dataset

### Neural network

- ```lr```: the learning rate is the step size used in the optimisation procedure when training the network to reach the local minima of the loss function assigned during training 
- ```epochs```: the numbes of full dataset passes to be run during the training procedure

### Convolutional neural network

- ```lr```: the learning rate is the step size used in the optimisation procedure when training the network to reach the local minima of the loss function assigned during training 
- ```epochs```: the number of full dataset passes to be run during the training procedure

### Geometric neural network

- ```lr```: the learning rate is the step size used in the optimisation procedure when training the network to reach the local minima of the loss function assigned during training 
- ```epochs```: the number of full dataset passes to be run during the training procedure

### Running code

Before runnning the code steps below ensure that one has navigated to the data project's directory: 

1. Initialise the Python virtual environment as guided in [Python virtual environment](#python-virtual-environment)
2. Set up configuration as desired to the user's purposes using the configuration JSON file as described in [Configuration](#configuration) and setting the relevant hyperparameters as described in [Hyperparameters](#hyperparameters)
3. Change directories such that the user is in the [source code](src/) directory
4. Run the command ```python3 main.py``` to execute the main programme
5. The programme will output as it progresses through the programme updating the user as to what it completed and its progress through the different models' training procedures
6. Once the programme has outputed ```Completed programme``` all the outputs detailed in [Outputs](#outputs) will be saved and ready for analysis


## Outputs

An example of the programme logging outputs produed by ```main.py``` can be viewed from this [logged file](outputs/programme-outputs/cli_outputs_example.txt). This is not a standard output of the programme, but simply added to the repository for completeness and to aid end users.

### Model parameters

The model parameters or models themselves for each of the methods are saved in the [saved-models](outputs/saved-models):

1. Gaussian naive-Bayes: the mean and standard deviation for each category belonging to the dataset are saved as model parameters to a [parameters](outputs/saved-models/example_1_gaussian_naive_bayes_model_parameters.txt) file, the ```smoothing``` hyperparameter is also saved for completeness and potential later use
2. $k$-Nearest neighbours: the value for the optimum $k$ within the range of ```kmin``` and ```kmax```, and the number of splits used for cross-validation during the optimisation process are saved as hyperparameters to the [parameters](outputs/saved-models/example_1_knn_model_parameters.txt) file for completeness and potential later use
3. Neural network: the model was implemented in PyTorch and so the entire trained model is saved as a ```.pth``` [file](outputs/saved-models/example_2_nn_model.pth) for completeness and potential later use
4. Convolutional neural network: the model was implemented in PyTorch and so the entire trained model is saved as a ```.pth``` [file](outputs/saved-models/example_2_cnn_model.pth) for completeness and potential later use
5. Graph neural network: the model was implemented in PyTorch and so the entire trained model is saved as a ```.pth``` [file](outputs/saved-models/example_1_gnn_model.pth) for completeness and potential later use

### Model leanring curves

The learning process for each of the methods is saved and the training progress can be viewed via the training curve graphs that are plotted in the [learning-curves](outputs/learning-curves/) folder. The Gaussian naive-Bayes does not have a learning curve output because there was no iterative process in this algorithm which was implemented. The $k$-Nearest neighbours algorithm has a curve displaying the loss against $k$. This is strictly not a training curve but a grid search exercise to find the optimum $k$ for the given dataset. The three deep learning methods all have the classic training curves which can be further understood from the [Background knowledge](#background-knowledge).

### Model predictions

The main programme, ```main.py```, will produce a single prediction for each of the methods where a selected test image from the given dataset is hard coded into the programme for each method and the image is saved with the image, image's true classification, and its predicted classification. Examples of the model predictions output can be found in the [test-predictions](outputs/test-predictions/) folder.


## Background knowledge

This section is to provide the end users with resources for understanding the theory behind the implementations of the methods used in this project. For each of the methods background theory resources are provided in the form of papers, articles, and textbooks; some alternative code implementations where possible; and in the case of the graph neural network previous implementations of the class from the presented article.

### Gaussian naive-Bayes

- **Textbook Chapters:**
  - Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media. Chapter 2.
  - Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.2.

- **Papers:**
  - Langley, P., Iba, W., & Thompson, K. (1992). *An Analysis of Bayesian Classifiers*. In *AAAI* (pp. 223-228).

- **Alternative Implementation:**
  - [Scikit-learn GaussianNB Implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/naive_bayes.py)


### $k$-Nearest neighbours

- **Textbook Chapters:**
  - Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 2.5.
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 13.3.

- **Papers:**
  - Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification*. *IEEE Transactions on Information Theory*, 13(1), 21-27.

- **Alternative Implementation:**
  - [KNN-python-implementation](https://github.com/chingisooinar/KNN-python-implementation)

### Neural Network
- **Textbook Chapters:**
  - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.
  - Nielsen, M. (2015). *Neural Networks and Deep Learning*. Chapter 1.

- **Papers:**
  - Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning Representations by Back-Propagating Errors*. *Nature*, 323(6088), 533-536.

- **Alternative Implementation:**
  - [PyTorch NN example](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb)

### Convolutional Neural Network
- **Textbook Chapters:**
  - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 9.
  - Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media. Chapter 14.

- **Papers:**
  - LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989) "Backpropagation Applied to Handwritten Zip Code Recognition", AT&T Bell Laboratories.

- **Alternative Implementation:**
  - [PyTorch CNN example](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)

### Graph Neural Network
- **Textbook Chapters:**
  - Hamilton, W. L. (2020). *Graph Representation Learning*. Morgan & Claypool Publishers. Chapters 1-3.

- **Papers:**
  - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. arXiv:1711.08920.
  - Kipf, T. N., & Welling, M. (2016). *Semi-Supervised Classification with Graph Convolutional Networks*. *arXiv preprint arXiv:1609.02907*.

- **Articles**
    - [Tutorial on Graph Neural Networks for Computer Vision and Beyond](https://medium.com/@BorisAKnyazev/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-1-3d9fada3b80d)

- **Implementation followed from graph neural network article:**
  - [PyTorch Graph Neural Network MNIST](https://github.com/DebasmitaGhose/PyTorch_Graph_Neural_Network_MNIST/tree/master)