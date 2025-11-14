"""
Models module for MNIST digit recognition.

This module provides a CNN architecture for classifying MNIST handwritten digits.

The model (MNISTConvNet) uses adaptive pooling to handle any input resolution,
making it suitable for real-world applications where images may not be exactly 28x28.
"""

from .cnn import MNISTConvNet

__all__ = ["MNISTConvNet"]
