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

# Custom modules
from logger import logProgress

#=============================================================================
# Functions
#=============================================================================
    

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
metadata_file_path = "configurations.json"

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
    with open(metadata_file_path, "r") as metadata_file:
        json_data = json.load(metadata_file)
    logProgress("Imported metadata...")
    
    # Setting metadata variables
    loggingName     = json_data["loggingName"]
    runNumber       = json_data["runNumber"]
    dataPath        = json_data["dataPath"]
    outputFigPath   = json_data["outputFigPath"]
    outputValPath   = json_data["outputValPath"]
    outputModelPath = json_data["outputModelPath"]
    
    # Update metadata file for next run
    json_data["runNumber"] = str(int(runNumber) + 1)
    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(json_data, metadata_file, indent=4)
    
    # Loading data

    
    #==========================================================================
    # Data pre-processing
    #==========================================================================
    
    
    #==========================================================================
    # Programme completion
    #==========================================================================

    logProgress("Completed programme")