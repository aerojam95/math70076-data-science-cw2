#=============================================================================
# Programme: 
# Data loading function for the main programme
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

import numpy as np
import gzip

#=============================================================================
# Functions
#=============================================================================

def loadImages(file_path:str):
    """
    function that returns the loaded images from a .gz file

    Args:
        file_path (str): file path for the .gz image file to be loaded 

    Returns:
        data (np array): loaded image data 
    """
    with gzip.open(file_path, 'rb') as f:
        # Skip the magic number and dimensions (first 16 bytes)
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)  # reshape to image size

def loadLabels(file_path:str):
    """
    function that returns the loaded labels from a .gz file

    Args:
        file_path (str): file path for the .gz label file to be loaded

    Returns:
        data (np array): loaded label data 
    """
    with gzip.open(file_path, 'rb') as f:
        # Skip the magic number (first 8 bytes)
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data