"""Logging utilities."""

import numpy as np


def memory_usage(array: np.ndarray) -> str:
    """Returns the memory usage of a numpy array in a human-readable format."""
    nbytes = array.nbytes
    if nbytes < 1024:
        return f"{nbytes} bytes"
    elif nbytes < 1024**2:
        return f"{nbytes / 1024:.2f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / 1024**2:.2f} MB"
    else:
        return f"{nbytes / 1024**3:.2f} GB"
