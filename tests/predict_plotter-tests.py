#=============================================================================
# Unit test:
# Data loading function unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
import torch
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Custom modules
from predict_plotter import predictPlotter

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class for logProgress.py
#=============================================================================

class TestPredictPLotterFunction(unittest.TestCase):
    def test_predictPlotter(self):
        # Define some sample data
        image = torch.rand(28, 28)  # Sample image tensor
        trueLabel = 3
        predictedLabel = 5
        classes = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
        file = "test_plot.png"
        
        # Call the function
        predictPlotter(image, trueLabel, predictedLabel, classes, file)
        
        # Check if the file is created
        assert os.path.exists(file), "Plot file does not exist"
        
        self.assertTrue(os.path.exists(file))
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error: {e.strerror}")

    
if __name__ == "__main__":
    unittest.main()