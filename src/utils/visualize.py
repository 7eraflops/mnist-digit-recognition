"""
Visualization utilities for MNIST digit recognition.

This module provides functions for visualizing training progress, model predictions,
confusion matrices, and sample images from the dataset.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary containing training history with keys:
                 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(
        epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2
    )
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(
            epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2
        )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(
        epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2
    )
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(
            epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2
        )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title(
        "Training and Validation Accuracy", fontsize=14, fontweight="bold"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Plot confusion matrix for model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (default: ['0', '1', ..., '9'])
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        normalize: Whether to normalize the confusion matrix

    Returns:
        np.ndarray: Confusion matrix
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Convert tensors to numpy
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    # Compute confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return cm


def plot_sample_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot sample predictions from the model.

    Args:
        model: Trained model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    model.eval()

    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)

    # Select samples to display
    num_samples = min(num_samples, len(images))
    rows = (num_samples + 4) // 5
    cols = min(5, num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.5 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_samples):
        img = images[idx].cpu().numpy().squeeze()
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        confidence = probabilities[idx][pred_label].item() * 100

        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis("off")

        # Color code: green for correct, red for incorrect
        color = "green" if true_label == pred_label else "red"
        title = f"True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%"
        axes[idx].set_title(title, fontsize=9, color=color)

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sample predictions saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_misclassified_samples(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot misclassified samples from the model.

    Args:
        model: Trained model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        num_samples: Number of misclassified samples to display
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    model.eval()

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_probs = []

    # Collect misclassified samples
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)

            # Find misclassified samples
            mask = predicted != labels
            if mask.sum() > 0:
                misclassified_images.append(images[mask].cpu())
                misclassified_labels.append(labels[mask].cpu())
                misclassified_preds.append(predicted[mask].cpu())
                misclassified_probs.append(probabilities[mask].cpu())

            if sum([len(x) for x in misclassified_images]) >= num_samples:
                break

    if len(misclassified_images) == 0:
        print("No misclassified samples found!")
        return

    # Concatenate all misclassified samples
    misclassified_images = torch.cat(misclassified_images)[:num_samples]
    misclassified_labels = torch.cat(misclassified_labels)[:num_samples]
    misclassified_preds = torch.cat(misclassified_preds)[:num_samples]
    misclassified_probs = torch.cat(misclassified_probs)[:num_samples]

    # Plot
    num_samples = len(misclassified_images)
    rows = (num_samples + 4) // 5
    cols = min(5, num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.5 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_samples):
        img = misclassified_images[idx].numpy().squeeze()
        true_label = misclassified_labels[idx].item()
        pred_label = misclassified_preds[idx].item()
        confidence = misclassified_probs[idx][pred_label].item() * 100

        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis("off")
        title = f"True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%"
        axes[idx].set_title(title, fontsize=9, color="red")

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Misclassified Samples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Misclassified samples saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot learning rate schedule over epochs.

    Args:
        history: Dictionary containing 'learning_rate' history
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if "learning_rate" not in history or len(history["learning_rate"]) == 0:
        print("No learning rate history available")
        return

    epochs = range(1, len(history["learning_rate"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["learning_rate"], "b-", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning rate plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_classification_report(
    y_true: torch.Tensor, y_pred: torch.Tensor, class_names: Optional[List[str]] = None
) -> None:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (default: ['0', '1', ..., '9'])
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Convert tensors to numpy
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true_np, y_pred_np, target_names=class_names))


def visualize_dataset_samples(
    data_loader: torch.utils.data.DataLoader,
    num_samples: int = 25,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize random samples from the dataset.

    Args:
        data_loader: DataLoader containing data
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    num_samples = min(num_samples, len(images))
    rows = (num_samples + 4) // 5
    cols = min(5, num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_samples):
        img = images[idx].numpy().squeeze()
        label = labels[idx].item()

        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Label: {label}", fontsize=10)

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Dataset Samples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Dataset samples saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
