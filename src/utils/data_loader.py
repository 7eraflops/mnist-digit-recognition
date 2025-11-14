"""
Data loading and preprocessing utilities for MNIST dataset.

This module handles downloading, loading, and preprocessing the MNIST dataset
with support for data augmentation and custom transforms.
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataLoader:
    """
    Handles MNIST dataset loading and preprocessing.

    This class manages dataset downloading, splitting, and creating DataLoader
    objects for training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        val_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
        augment: bool = False,
    ):
        """
        Initialize the MNIST data loader.

        Args:
            data_dir (str): Directory to store/load MNIST data
            batch_size (int): Batch size for data loaders
            val_split (float): Fraction of training data to use for validation
            num_workers (int): Number of workers for data loading
            seed (int): Random seed for reproducibility
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.augment = augment

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

    def get_transforms(self, train: bool = True) -> transforms.Compose:
        """
        Get data transformations for training or testing.

        Args:
            train (bool): Whether to get training transforms (with augmentation)

        Returns:
            transforms.Compose: Composed transforms
        """
        if train and self.augment:
            # Training transforms with data augmentation
            transform = transforms.Compose(
                [
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            # Standard transforms for validation/testing
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

        return transform

    def load_data(
        self,
    ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """
        Load and prepare MNIST data loaders.

        Returns:
            Tuple[DataLoader, Optional[DataLoader], DataLoader]:
                - train_loader: DataLoader for training data
                - val_loader: DataLoader for validation data (None if val_split=0)
                - test_loader: DataLoader for test data
        """
        # Download and load training data
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.get_transforms(train=True),
        )

        # Download and load test data
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.get_transforms(train=False),
        )

        # Split training data into train and validation sets
        val_loader = None
        if self.val_split > 0:
            val_size = int(len(train_dataset) * self.val_split)
            train_size = len(train_dataset) - val_size

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Create validation loader
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        # Create training loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def get_data_info(self):
        """
        Get information about the MNIST dataset.

        Returns:
            dict: Dictionary containing dataset information
        """
        # Load a sample to get dataset info
        sample_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True)

        return {
            "num_classes": 10,
            "input_shape": (1, 28, 28),
            "train_samples": len(sample_dataset),
            "mean": 0.1307,
            "std": 0.3081,
        }


def create_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Convenience function to create MNIST data loaders.

    Args:
        data_dir (str): Directory to store/load MNIST data
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        augment (bool): Whether to apply data augmentation

    Returns:
        Tuple[DataLoader, Optional[DataLoader], DataLoader]:
            Training, validation (optional), and test data loaders
    """
    loader = MNISTDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        augment=augment,
    )
    return loader.load_data()
