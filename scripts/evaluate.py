#!/usr/bin/env python3
"""
Evaluation script for MNIST digit recognition.

This script evaluates a trained model on the test dataset and generates
comprehensive performance metrics and visualizations.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MNISTConvNet
from src.utils import (
    Trainer,
    create_data_loaders,
    plot_confusion_matrix,
    plot_misclassified_samples,
    plot_sample_predictions,
    print_classification_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST digit recognition model"
    )

    # Model parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (must match training)",
    )

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing MNIST data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def get_device(device_str=None):
    """Get the device to use for evaluation."""
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def create_model(args):
    """Create model based on arguments."""
    model = MNISTConvNet(dropout_rate=args.dropout)
    print("Using MNISTConvNet (adaptive CNN with multi-resolution support)")
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("\nEvaluating model on test set...")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Calculate accuracy
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            total_loss += loss.item()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Store predictions
            all_predictions.append(predicted.cpu())
            all_labels.append(target.cpu())
            all_probabilities.append(probabilities.cpu())

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)

    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities


def save_results(model, test_loader, device, predictions, labels, args):
    """Save evaluation results and visualizations."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # Plot confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
        show=False,
    )

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        normalize=True,
        save_path=os.path.join(args.output_dir, "confusion_matrix_normalized.png"),
        show=False,
    )

    # Print and save classification report
    print_classification_report(labels, predictions)

    # Plot sample predictions
    plot_sample_predictions(
        model,
        test_loader,
        device,
        num_samples=25,
        save_path=os.path.join(args.output_dir, "sample_predictions.png"),
        show=False,
    )

    # Plot misclassified samples
    plot_misclassified_samples(
        model,
        test_loader,
        device,
        num_samples=25,
        save_path=os.path.join(args.output_dir, "misclassified_samples.png"),
        show=False,
    )

    print(f"\nEvaluation results saved to {args.output_dir}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Get device
    device = get_device(args.device)

    # Load test data
    print("Loading MNIST test dataset...")
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.0,  # No validation split needed for evaluation
        num_workers=args.num_workers,
        seed=args.seed,
        augment=False,
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Create and load model
    print("\nLoading model...")
    model = create_model(args)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Evaluate model
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    test_loss, test_acc, predictions, labels, probabilities = evaluate_model(
        model, test_loader, device
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Error Rate: {100 - test_acc:.2f}%")

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for digit in range(10):
        mask = labels == digit
        if mask.sum() > 0:
            digit_acc = (
                100.0 * (predictions[mask] == labels[mask]).sum().item() / mask.sum()
            )
            print(f"  Digit {digit}: {digit_acc:.2f}%")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    save_results(model, test_loader, device, predictions, labels, args)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
