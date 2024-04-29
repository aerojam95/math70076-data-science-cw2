#=============================================================================
# Unit test:
# logging function unit test
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import unittest
from unittest.mock import patch
from io import StringIO
import re
import sys
import os

# Append the path of `src` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Custom modules
from logger import logProgress

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Unit trst class for logProgress.py
#=============================================================================

class TestLogProgress(unittest.TestCase):
    @patch('sys.stdout', new_callable=StringIO)
    def test_log_progress_default_format(self, mock_stdout):
        test_message = "Test message"
        logProgress(test_message)
        output = mock_stdout.getvalue()
        self.assertIn(test_message, output)
        pattern = r'\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}: ' + re.escape(test_message)
        self.assertTrue(re.match(pattern, output), "Output does not match the expected format")

if __name__ == '__main__':
    unittest.main()