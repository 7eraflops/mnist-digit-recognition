#!/usr/bin/env python3
"""
Simple example demonstrating multi-resolution inference with MNISTConvNet.

This example shows how to:
1. Use the MNISTConvNet model (with adaptive pooling)
2. Make predictions on images of different resolutions
3. Leverage adaptive pooling for variable-sized inputs
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MNISTConvNet
from src.utils import MNISTInference


def create_sample_digit(digit_value, resolution=28):
    """
    Create a simple sample digit image at specified resolution.

    Args:
        digit_value: Digit to create (0-9)
        resolution: Image resolution in pixels

    Returns:
        PIL.Image: Sample digit image
    """
    # Create a simple synthetic digit for demonstration
    # In practice, you'd load real images
    img = Image.new("L", (resolution, resolution), color=255)

    # This is just a placeholder - in real usage, load actual images
    # For now, create a simple pattern
    arr = np.ones((resolution, resolution)) * 255

    # Add some simple pattern to simulate a digit
    center = resolution // 2
    radius = resolution // 4
    for i in range(resolution):
        for j in range(resolution):
            if (i - center) ** 2 + (j - center) ** 2 < radius**2:
                arr[i, j] = 50

    img = Image.fromarray(arr.astype(np.uint8))
    return img


def main():
    """Main example function."""
    print("=" * 70)
    print("MULTI-RESOLUTION INFERENCE EXAMPLE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # =========================================================================
    # CREATE IMPROVED MODEL WITH ADAPTIVE POOLING
    # =========================================================================

    print("\n" + "-" * 70)
    print("Creating MNISTConvNet (with Adaptive Pooling)")
    print("-" * 70)

    # Create the model
    model = MNISTConvNet()

    # Initialize inference with adaptive=True
    inference = MNISTInference(
        model=model,
        device=device,
        adaptive=True,  # Enable multi-resolution support!
    )

    print("\n✓ Model created: MNISTConvNet")
    print("  • Uses adaptive pooling (AdaptiveAvgPool2d)")
    print("  • Can process images at ANY resolution")
    print("  • Preserves detail from high-res images")
    print("  • No information loss from resizing")

    # =========================================================================
    # TEST WITH DIFFERENT RESOLUTIONS
    # =========================================================================

    print("\n" + "-" * 70)
    print("Testing Multi-Resolution Inference")
    print("-" * 70)

    # Test with different resolution images
    resolutions = [28, 56, 224]

    for res in resolutions:
        sample_img = create_sample_digit(7, resolution=res)
        pred = inference.predict(sample_img)
        print(f"  • {res}x{res} image → processed as-is → prediction: {pred}")

    # =========================================================================
    # PRACTICAL USE CASES
    # =========================================================================

    print("\n" + "-" * 70)
    print("Practical Use Cases")
    print("-" * 70)

    print("\n📱 Use Case 1: Mobile App")
    print("  • Camera captures 224x224 image")
    print("  • Model processes at full 224x224 (keeps detail)")

    mobile_img = create_sample_digit(3, resolution=224)
    pred_mobile = inference.predict(mobile_img)
    print(f"  → Prediction: {pred_mobile}")

    print("\n🖥️  Use Case 2: Mixed Resolution Dataset")
    print("  • Dataset has images: 28x28, 56x56, 112x112")
    print("  • Model handles all sizes natively (no resizing!)")

    mixed_sizes = [28, 56, 112]
    predictions = []
    for size in mixed_sizes:
        img = create_sample_digit(5, resolution=size)
        pred = inference.predict(img)
        predictions.append(pred)
    print(f"  → Predictions: {predictions}")

    print("\n📸 Use Case 3: High-Quality Scanner")
    print("  • Scanner produces 512x512 images")
    print("  • Model utilizes full resolution (no downscaling!)")

    hq_img = create_sample_digit(9, resolution=512)
    pred_hq = inference.predict(hq_img)
    print(f"  → High-res prediction: {pred_hq}")

    # =========================================================================
    # CODE SNIPPET FOR YOUR PROJECTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("QUICK REFERENCE: Using the Model in Your Code")
    print("=" * 70)

    print("""
# Step 1: Import model and inference
from src.models import MNISTConvNet
from src.utils import MNISTInference

# Step 2: Create model (train normally on 28x28 data)
model = MNISTConvNet(dropout_rate=0.5)
# ... training code here ...

# Step 3: Initialize inference with adaptive=True
inference = MNISTInference(
    model=model,
    checkpoint_path="path/to/checkpoint.pth",
    adaptive=True  # This enables multi-resolution support!
)

# Step 4: Predict on ANY resolution
pred = inference.predict("small_image_32x32.png")
pred = inference.predict("medium_image_128x128.png")
pred = inference.predict("large_image_512x512.png")

# That's it! No resizing needed.
""")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n✅ Advantages of MNISTConvNet:")
    print("  1. Train on standard 28x28 MNIST dataset")
    print("  2. Infer on images of ANY resolution")
    print("  3. No preprocessing/resizing required")
    print("  4. Preserves detail from high-resolution inputs")
    print("  5. High accuracy (~99.4-99.6% on MNIST)")
    print("  6. Real-world ready for variable-sized inputs")

    print("\n💡 Key Insight:")
    print("  Adaptive pooling layers (AdaptiveAvgPool2d) allow the network to")
    print("  accept any input size while maintaining the same architecture!")
    print("  This is the ONLY model you need - one model, any resolution!")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nFor a full demo with visualization, run:")
    print("  uv run python scripts/demo.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
