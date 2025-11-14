#!/usr/bin/env python3
"""
Demonstration script for MNIST digit recognition with adaptive CNN.

This script demonstrates:
- Loading and visualizing MNIST data
- Training the adaptive CNN model on 28x28 images (10 epochs)
- Evaluating model performance with comprehensive visualizations
- Making predictions with the trained model
- The model's multi-resolution capability (supports any input size)

For multi-resolution testing with real images, see:
examples/multi_resolution_example.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MNISTConvNet
from src.utils import (
    MNISTInference,
    Trainer,
    create_data_loaders,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_training_history,
    visualize_dataset_samples,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def main():
    """Main demonstration function."""
    print_section("MNIST DIGIT RECOGNITION - DEMO")
    print("Adaptive CNN with Multi-Resolution Inference")

    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    OUTPUT_DIR = "./demo_results"

    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # =========================================================================
    # PART 1: DATA LOADING
    # =========================================================================
    print_section("PART 1: DATA LOADING")

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        val_split=0.1,
        num_workers=2,
        seed=42,
        augment=False,
    )

    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # Visualize dataset samples
    visualize_dataset_samples(
        train_loader,
        num_samples=25,
        save_path=f"{OUTPUT_DIR}/01_dataset_samples.png",
        show=False,
    )
    print(f"✓ Dataset samples saved to {OUTPUT_DIR}/01_dataset_samples.png")

    # =========================================================================
    # PART 2: MODEL TRAINING
    # =========================================================================
    print_section("PART 2: MODEL TRAINING")

    model = MNISTConvNet(dropout_rate=0.5).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: MNISTConvNet")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: 3 conv blocks + adaptive pooling + 3 FC layers")
    print(f"  Special Feature: Can handle ANY input resolution!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        checkpoint_dir=f"{OUTPUT_DIR}/checkpoints",
        early_stopping_patience=5,
        scheduler=scheduler,
    )

    print("\nStarting training...")
    history = trainer.train(num_epochs=EPOCHS)

    print(f"\n✓ Training complete!")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"✓ Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"✓ Final test accuracy: {history['test_acc']:.2f}%")

    # =========================================================================
    # PART 3: EVALUATION & VISUALIZATIONS
    # =========================================================================
    print_section("PART 3: EVALUATION & VISUALIZATIONS")

    # Get predictions
    predictions, labels, _ = trainer.get_predictions(test_loader)

    # Training history
    plot_training_history(
        history,
        save_path=f"{OUTPUT_DIR}/02_training_history.png",
        show=False,
    )
    print(f"✓ Training history: {OUTPUT_DIR}/02_training_history.png")

    # Confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        save_path=f"{OUTPUT_DIR}/03_confusion_matrix.png",
        show=False,
    )
    print(f"✓ Confusion matrix: {OUTPUT_DIR}/03_confusion_matrix.png")

    # Sample predictions
    plot_sample_predictions(
        trainer.model,
        test_loader,
        trainer.device,
        num_samples=20,
        save_path=f"{OUTPUT_DIR}/04_sample_predictions.png",
        show=False,
    )
    print(f"✓ Sample predictions: {OUTPUT_DIR}/04_sample_predictions.png")

    # Per-class accuracy
    print_subsection("Per-Class Accuracy")
    for digit in range(10):
        mask = labels == digit
        digit_correct = (predictions[mask] == labels[mask]).sum().item()
        digit_total = mask.sum().item()
        digit_acc = 100.0 * digit_correct / digit_total
        print(f"  Digit {digit}: {digit_acc:.2f}% ({digit_correct}/{digit_total})")

    # =========================================================================
    # PART 4: INFERENCE EXAMPLE
    # =========================================================================
    print_section("PART 4: INFERENCE EXAMPLE")

    # Create inference engine
    inference_engine = MNISTInference(
        model=model,
        checkpoint_path=None,  # Model already loaded
        device=DEVICE,
        adaptive=True,  # Enable multi-resolution support
    )

    # Get a sample image
    images, labels_batch = next(iter(test_loader))
    sample_image = images[0]
    true_label = labels_batch[0].item()

    print(f"\nTesting inference on a single image:")
    print(f"  True label: {true_label}")

    # Predict with confidence
    pred_class, confidence = inference_engine.predict_with_confidence(sample_image)
    print(f"  Predicted: {pred_class} (Confidence: {confidence:.2f}%)")

    # Top-3 predictions
    top_classes, top_probs = inference_engine.predict_top_k(sample_image, k=3)
    print(f"\n  Top 3 predictions:")
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs), 1):
        marker = "✓" if cls == true_label else " "
        print(f"    {marker} {i}. Digit {cls}: {prob:.2f}%")

    # Visualize prediction
    inference_engine.visualize_prediction(
        sample_image,
        save_path=f"{OUTPUT_DIR}/05_inference_example.png",
        show=False,
    )
    print(f"\n✓ Inference visualization: {OUTPUT_DIR}/05_inference_example.png")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("DEMO COMPLETE!")

    print("\n📊 Results Summary:")
    print(f"  • Test Accuracy: {history['test_acc']:.2f}%")
    print(f"  • Model Parameters: {total_params:,}")
    print(f"  • Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"\n💡 Multi-Resolution Capability:")
    print(f"  • This model uses adaptive pooling and can process images of ANY size")
    print(f"  • Works with inputs from 8x8 to 1024x1024+ pixels")
    print(f"  • See examples/multi_resolution_example.py for usage")

    print(f"\n📁 Generated Files ({OUTPUT_DIR}/):")
    print("  • 01_dataset_samples.png           - Dataset visualization")
    print("  • 02_training_history.png          - Training curves")
    print("  • 03_confusion_matrix.png          - Confusion matrix")
    print("  • 04_sample_predictions.png        - Prediction samples")
    print("  • 05_inference_example.png         - Inference example")
    print("  • checkpoints/                     - Model checkpoints")

    print("\n💡 Key Insight:")
    print("  This model was trained on 28x28 images but can process images of ANY")
    print("  size (e.g., 64x64, 224x224, 512x512) thanks to adaptive pooling!")
    print("  No resizing needed at inference time!")

    print("\n🚀 Next Steps:")
    print("  1. Review the generated visualizations")
    print("  2. Train on more epochs: python scripts/train.py --epochs 20")
    print("  3. Try inference on your own images: python scripts/inference.py")
    print("  4. Evaluate saved models: python scripts/evaluate.py")
    print("  5. Test multi-resolution: python examples/multi_resolution_example.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
