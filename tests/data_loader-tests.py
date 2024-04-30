#=============================================================================
# Unit test:
# Data loading function unit tests
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
from unittest.mock import patch, mock_open
from io import StringIO
import numpy as np
import gzip
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Custom modules
from data_loader import loadImages, loadLabels

#=============================================================================
# Functions
#=============================================================================

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit test class for logProgress.py
#=============================================================================

class TestLoadDataFunctions(unittest.TestCase):

    @patch('gzip.open')
    def testLoadImages(self, mock_gzip_open):
        # Create a mock to simulate gzip.open and np.frombuffer
        mock_gzip_open.return_value.__enter__.return_value.read.return_value = b'\x00' * 16 + b'\x03' * 28 * 28 * 10  # Simulate data for 10 images
        with patch('gzip.open', mock_gzip_open):
            images = loadImages('dummy_path.gz')
            self.assertEqual(images.shape, (10, 28, 28))  # 10 images, 28x28 pixels each
            self.assertTrue((images == 3).all())  # All values should be 3 based on mock data

    @patch('gzip.open')
    def testLoadLabels(self, mock_gzip_open):
        # Create a mock to simulate gzip.open and np.frombuffer
        mock_gzip_open.return_value.__enter__.return_value.read.return_value = b'\x00' * 8 + b'\x01' * 10  # Simulate data for 10 labels
        with patch('gzip.open', mock_gzip_open):
            labels = loadLabels('dummy_path.gz')
            self.assertEqual(len(labels), 10)  # Should be 10 labels
            self.assertTrue((labels == 1).all())  # All values should be 1 based on mock data

if __name__ == '__main__':
    unittest.main()