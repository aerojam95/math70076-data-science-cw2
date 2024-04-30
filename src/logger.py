#=============================================================================
# Programme: 
# logging functions for running the main programme
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

from datetime import datetime

#=============================================================================
# Functions
#=============================================================================

def logProgress(message: str, format: str = "%d %b %Y %H:%M:%S: "):
    """
    Prints a log message prefixed with the current datetime formatted according to the provided format string

    Args:
    - message (str): The log message to be printed
    - format (str, optional): The datetime format string. Defaults to "%d %b %Y %H:%M:%S: "

    Returns:
    - None. The function directly prints the formatted message to the standard output

    Example:
    >>> logProgress("Hello, world!")
    01 Jan 2000 00:00:00: Hello, world!  # Example output; actual output will vary based on the current datetime.
    """
    formattedMessage = datetime.now().strftime(format + message)
    print(formattedMessage)
    return None