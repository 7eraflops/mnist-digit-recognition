"""
Training utilities for MNIST digit recognition.

This module provides a Trainer class that handles model training, validation,
early stopping, learning rate scheduling, and checkpoint management.
"""

import os
import time
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Trainer class for managing model training and validation.

    This class handles the complete training pipeline including:
    - Training loop with progress tracking
    - Validation and early stopping
    - Learning rate scheduling
    - Checkpoint saving and loading
    - Training history logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        early_stopping_patience: int = 10,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (CPU/GPU)
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs to wait before early stopping
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.scheduler = scheduler

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # Early stopping variables
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.best_model_path = None

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple[float, float]: Average training loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on a given dataset.

        Args:
            loader: DataLoader for validation/test data

        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(loader, desc="Validating", leave=False):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            int: Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get("history", self.history)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from epoch {epoch}")

        return epoch

    def check_early_stopping(self, val_loss: float, val_acc: float, epoch: int) -> bool:
        """
        Check if early stopping criteria is met and save best model.

        Args:
            val_loss: Current validation loss
            val_acc: Current validation accuracy
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0

            # Save best model
            self.best_model_path = f"best_model_epoch_{epoch}.pth"
            self.save_checkpoint(epoch, self.best_model_path)
            print(f"New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            return False
        else:
            self.epochs_without_improvement += 1
            print(
                f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs"
            )

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                return True

            return False

    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume training from (optional)

        Returns:
            Dict: Training history
        """
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"\nTraining on device: {self.device}")
        print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Validation
            if self.val_loader is not None:
                val_loss, val_acc = self.validate(self.val_loader)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping check
                if self.check_early_stopping(val_loss, val_acc, epoch):
                    break

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            else:
                # No validation set - use training loss for scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_loss)
                    else:
                        self.scheduler.step()

                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}.pth")

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training completed in {total_time:.2f}s")

        # Final test evaluation
        if self.best_model_path is not None:
            print(f"\nLoading best model from {self.best_model_path}")
            checkpoint_path = os.path.join(self.checkpoint_dir, self.best_model_path)
            self.load_checkpoint(checkpoint_path)

        test_loss, test_acc = self.validate(self.test_loader)
        print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        self.history["test_loss"] = test_loss
        self.history["test_acc"] = test_acc

        return self.history

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Returns:
            Dict[str, float]: Test metrics
        """
        test_loss, test_acc = self.validate(self.test_loader)
        return {"test_loss": test_loss, "test_accuracy": test_acc}

    def get_predictions(
        self, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions for a dataset.

        Args:
            loader: DataLoader to get predictions for

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                predictions, true labels, and probabilities
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, preds = torch.max(output, 1)

                all_preds.append(preds.cpu())
                all_labels.append(target)
                all_probs.append(probs.cpu())

        return (
            torch.cat(all_preds),
            torch.cat(all_labels),
            torch.cat(all_probs),
        )
