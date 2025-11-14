#!/usr/bin/env python3
"""
Main training script for MNIST digit recognition.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training with validation
- Checkpoint saving
- Visualization of results
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MNISTConvNet
from src.utils import (
    Trainer,
    create_data_loaders,
    plot_confusion_matrix,
    plot_learning_rate,
    plot_misclassified_samples,
    plot_sample_predictions,
    plot_training_history,
    print_classification_report,
)
from src.utils.config_loader import get_config


def parse_args(config):
    """Parse command line arguments, using config for defaults."""
    parser = argparse.ArgumentParser(description="Train MNIST digit recognition model")

    # Config file and preset
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Preset to use from the configuration file",
    )

    # Set defaults from config
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    path_cfg = config.get("paths", {})
    hardware_cfg = config.get("hardware", {})

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default=data_cfg.get("data_dir", "./data"),
        help="Directory to store/load MNIST data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=data_cfg.get("batch_size", 64),
        help="Batch size for training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=data_cfg.get("val_split", 0.1),
        help="Fraction of training data for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=data_cfg.get("num_workers", 4),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=data_cfg.get("augment", False),
        help="Use data augmentation during training",
    )

    # Model parameters
    parser.add_argument(
        "--dropout",
        type=float,
        default=model_cfg.get("dropout_rate", 0.5),
        help="Dropout rate for regularization",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_cfg.get("epochs", 20),
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=train_cfg.get("learning_rate", 0.001),
        help="Initial learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=train_cfg.get("optimizer", "adam"),
        choices=["adam", "sgd", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=train_cfg.get("momentum", 0.9),
        help="Momentum for SGD optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=train_cfg.get("weight_decay", 1e-4),
        help="Weight decay (L2 penalty)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=train_cfg.get("scheduler", "plateau"),
        choices=["none", "step", "plateau", "cosine"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=train_cfg.get("early_stopping_patience", 10),
        help="Early stopping patience (epochs)",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=path_cfg.get("checkpoint_dir", "./checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=path_cfg.get("resume_from"),
        help="Path to checkpoint to resume from",
    )

    # Output parameters
    parser.add_argument(
        "--log-dir",
        type=str,
        default=path_cfg.get("log_dir", "./logs"),
        help="Directory to save logs and plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=data_cfg.get("seed", 42),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=hardware_cfg.get("device"),
        help="Device to use (cuda/cpu). Auto-detect if not specified",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def get_device(device_str=None):
    """Get the device to use for training."""
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


def create_optimizer(model, args):
    """Create optimizer based on arguments."""
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    print(f"Using {args.optimizer.upper()} optimizer with lr={args.lr}")
    return optimizer


def create_scheduler(optimizer, args, config):
    """Create learning rate scheduler based on arguments."""
    scheduler_name = args.scheduler
    if scheduler_name == "none":
        return None

    scheduler_cfg = config.get("scheduler", {}).get(scheduler_name, {})

    if scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg)
        print("Using StepLR scheduler")
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
        print("Using ReduceLROnPlateau scheduler")
    else:  # cosine
        if scheduler_cfg.get("T_max") is None:
            scheduler_cfg["T_max"] = args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg)
        print("Using CosineAnnealingLR scheduler")

    return scheduler


def save_results(trainer, args):
    """Save training results and visualizations."""
    os.makedirs(args.log_dir, exist_ok=True)

    # Plot training history
    plot_training_history(
        trainer.history,
        save_path=os.path.join(args.log_dir, "training_history.png"),
        show=False,
    )

    # Plot learning rate
    plot_learning_rate(
        trainer.history,
        save_path=os.path.join(args.log_dir, "learning_rate.png"),
        show=False,
    )

    # Get predictions for confusion matrix
    predictions, labels, probabilities = trainer.get_predictions(trainer.test_loader)

    # Plot confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        save_path=os.path.join(args.log_dir, "confusion_matrix.png"),
        show=False,
    )

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        normalize=True,
        save_path=os.path.join(args.log_dir, "confusion_matrix_normalized.png"),
        show=False,
    )

    # Print classification report
    print_classification_report(labels, predictions)

    # Plot sample predictions
    plot_sample_predictions(
        trainer.model,
        trainer.test_loader,
        trainer.device,
        num_samples=20,
        save_path=os.path.join(args.log_dir, "sample_predictions.png"),
        show=False,
    )

    # Plot misclassified samples
    plot_misclassified_samples(
        trainer.model,
        trainer.test_loader,
        trainer.device,
        num_samples=20,
        save_path=os.path.join(args.log_dir, "misclassified_samples.png"),
        show=False,
    )

    print(f"\nResults saved to {args.log_dir}")


def main():
    """Main training function."""
    # First, parse only the config file and preset to load the configuration
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default="configs/config.toml", help="Path to config file"
    )
    config_parser.add_argument("--preset", help="Name of the preset to use")
    config_args, _ = config_parser.parse_known_args()

    config = get_config(preset=config_args.preset, config_path=config_args.config)

    # Now, parse all arguments with defaults from the loaded config
    args = parse_args(config)

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    # Create data loaders
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\nInitializing model...")
    model = create_model(args)
    model = model.to(device)

    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = create_optimizer(model, args)

    # Create scheduler
    scheduler = create_scheduler(optimizer, args, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping,
        scheduler=scheduler,
    )

    # Train model
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = trainer.train(num_epochs=args.epochs, resume_from=args.resume)

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(trainer, args)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Final test loss: {history['test_loss']:.4f}")
    print(f"Final test accuracy: {history['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
