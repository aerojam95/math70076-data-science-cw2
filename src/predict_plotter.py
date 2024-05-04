#=============================================================================
# Programme: 
# Prediction plotting function for the main programme
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

import matplotlib.pyplot as plt

#=============================================================================
# Functions
#=============================================================================

def predictPlotter(image, trueLabel:int, predictedLabel:int, classes:dict, FigPath:str):
    """
    Displays an image along with its true and predicted labels, and saves the plot to a file

    Args:
        image (): The image tensor
        trueLabel (int): The true label of the image
        predictedLabel (int): The predicted label of the image
        classes (dict): A dictionary mapping label indices to class names
        FigPath (str): The file path where the plot will be saved

    Returns:
        None
    """
    trueClass      = classes[trueLabel] 
    predoctedClass = classes[predictedLabel]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"True label: {trueClass}, Predicted label: {predoctedClass}")
    plt.savefig(FigPath)
    plt.close()
    return None