"""
MNIST Digit Recognition Package

A PyTorch-based implementation for recognizing handwritten digits
from the MNIST dataset using Convolutional Neural Networks.
"""

__version__ = "0.1.0"
__author__ = "MNIST Recognition Team"

from . import models
from . import utils

__all__ = ["models", "utils"]
