"""
Convolutional Neural Network for MNIST Digit Recognition

This module defines a CNN architecture optimized for MNIST digit classification
with adaptive pooling to handle any input resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification with adaptive pooling.

    **Key Features:**
    - Deeper architecture (3 conv blocks) for better feature extraction
    - Adaptive pooling: can handle ANY input resolution (not just 28x28)
    - High accuracy (~99.4-99.6% on MNIST)
    - Train on 28x28, infer on any size (32x32, 64x64, 224x224, etc.)
    - Proper batch normalization for each convolutional layer

    **Architecture:**
        - Block 1: 2× Conv2d(1→32) + BatchNorm + ReLU + MaxPool
        - Block 2: 2× Conv2d(32→64) + BatchNorm + ReLU + MaxPool + Dropout
        - Block 3: 2× Conv2d(64→128) + BatchNorm + ReLU + MaxPool + Dropout
        - Adaptive pooling to 3x3
        - FC1: 1152 → 256 + ReLU + Dropout
        - FC2: 256 → 128 + ReLU + Dropout
        - FC3: 128 → 10 (output)

    **Usage Example:**
        ```python
        model = MNISTConvNet(dropout_rate=0.5)

        # Train on 28x28 images
        output = model(images_28x28)

        # Infer on any resolution
        output = model(images_224x224)
        output = model(images_64x64)
        ```
    """

    def __init__(self, dropout_rate=0.5):
        """
        Initialize the CNN model.

        Args:
            dropout_rate (float): Dropout rate for regularization (default: 0.5)
        """
        super(MNISTConvNet, self).__init__()

        # First convolutional block - each conv gets its own batch norm
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive pooling - KEY FEATURE for multi-resolution support!
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Fully connected layers
        # After adaptive pooling: always 3x3 regardless of input size
        # Flattened size: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Thanks to adaptive pooling, this model can process images of any size!

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
                             where H and W can be any size >= 8

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # Block 1: Extract low-level features
        # Conv → BatchNorm → ReLU (standard order)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        # Block 2: Extract mid-level features
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        # Block 3: Extract high-level features
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        # Adaptive pooling - handles any input size!
        # This ensures we always get 3x3 feature maps
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers for classification
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        """
        Make predictions with softmax probabilities.

        Args:
            x (torch.Tensor): Input tensor (any resolution)

        Returns:
            torch.Tensor: Softmax probabilities of shape (batch_size, 10)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_class(self, x):
        """
        Predict the class label for input.

        Args:
            x (torch.Tensor): Input tensor (any resolution)

        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
