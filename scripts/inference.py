#!/usr/bin/env python3
"""
Inference script for MNIST digit recognition.

This script allows making predictions on individual images or batches of images
using a trained model.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MNISTConvNet
from src.utils import MNISTInference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions on MNIST digit images"
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
    parser.add_argument(
        "--force-original-size",
        action="store_true",
        help="Force processing at original size (skip auto-resize)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=28,
        help="Target size for resizing (default: 28 for MNIST)",
    )

    # Input parameters
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for prediction",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing multiple images",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./predictions",
        help="Directory to save prediction visualizations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show top-k predictions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified",
    )

    return parser.parse_args()


def get_device(device_str=None):
    """Get the device to use for inference."""
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
    return model


def should_resize(image_path, target_size=28, tolerance=8):
    """
    Determine if image should be resized based on its dimensions.

    Args:
        image_path: Path to image
        target_size: Target dimension (default: 28 for MNIST)
        tolerance: Pixels of tolerance (default: 8)

    Returns:
        tuple: (should_resize, width, height)
    """
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size

    # If image is within tolerance of target_size x target_size, keep original
    min_size = target_size - tolerance
    max_size = target_size + tolerance

    close_to_target = min_size <= width <= max_size and min_size <= height <= max_size

    return not close_to_target, width, height


def predict_single_image(
    inference_engine, image_path, args, was_resized=False, original_size=None
):
    """Make prediction on a single image."""
    from PIL import Image

    print(f"\nProcessing: {image_path}")
    if was_resized and original_size:
        print(
            f"  (resized from {original_size[0]}x{original_size[1]} to {args.target_size}x{args.target_size})"
        )
    print("-" * 50)

    # Save resized image if it was resized
    if was_resized and args.visualize:
        img = Image.open(image_path).convert("L")
        img_resized = img.resize((args.target_size, args.target_size), Image.BILINEAR)

        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"resized_{Path(image_path).stem}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        img_resized.save(output_path)
        print(f"Resized image saved to: {output_path}")

    # Get top-k predictions
    top_classes, top_probs = inference_engine.predict_top_k(image_path, k=args.top_k)

    # Print results
    print(f"Top-{args.top_k} predictions:")
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs), 1):
        print(f"  {i}. Digit {cls}: {prob:.2f}%")

    # Get prediction with confidence
    predicted_class, confidence = inference_engine.predict_with_confidence(image_path)
    print(
        f"\nFinal prediction: Digit {predicted_class} (Confidence: {confidence:.2f}%)"
    )

    # Create visualization if requested
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"prediction_{Path(image_path).stem}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        inference_engine.visualize_prediction(
            image_path,
            save_path=output_path,
            show=False,
        )
        print(f"Prediction visualization saved to: {output_path}")


def predict_image_directory(inference_engine, image_dir, args):
    """Make predictions on all images in a directory."""
    # Get list of image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"\nFound {len(image_files)} images in {image_dir}")
    print("=" * 50)

    # Make predictions on batch
    predictions = inference_engine.predict_batch(image_files)

    # Print results
    for image_path, pred in zip(image_files, predictions):
        filename = Path(image_path).name
        print(f"{filename}: Digit {pred}")

    # Create individual visualizations if requested
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nCreating visualizations...")

        for image_path in image_files:
            output_filename = f"prediction_{Path(image_path).stem}.png"
            output_path = os.path.join(args.output_dir, output_filename)

            inference_engine.visualize_prediction(
                image_path,
                save_path=output_path,
                show=False,
            )

        print(f"Visualizations saved to: {args.output_dir}")


def main():
    """Main inference function."""
    args = parse_args()

    # Validate arguments
    if args.image is None and args.image_dir is None:
        print("Error: Either --image or --image-dir must be specified")
        return

    if args.image is not None and args.image_dir is not None:
        print("Error: Cannot specify both --image and --image-dir")
        return

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    # Get device
    device = get_device(args.device)

    # Create model
    print("Loading model...")
    model = create_model(args)

    # Determine if we should resize based on image
    needs_resize = False
    original_size = None

    if args.image and not args.force_original_size:
        needs_resize, width, height = should_resize(
            args.image, target_size=args.target_size
        )
        original_size = (width, height)

    # Create inference engine
    # Use adaptive=True (keeps original size) if forcing original or image is close to target
    # Use adaptive=False (resizes) if image is far from target size
    adaptive = args.force_original_size or not needs_resize

    inference_engine = MNISTInference(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        adaptive=adaptive,
        target_size=(args.target_size, args.target_size),
    )

    print("Model loaded successfully!")
    if args.force_original_size:
        print("Processing at original image size (--force-original-size)")
    elif not needs_resize:
        print(
            f"Image size close to {args.target_size}x{args.target_size} - processing at original size"
        )
    else:
        print(
            f"Auto-resizing to {args.target_size}x{args.target_size} for better accuracy"
        )

    # Make predictions
    if args.image is not None:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return

        predict_single_image(
            inference_engine,
            args.image,
            args,
            was_resized=needs_resize,
            original_size=original_size,
        )

    else:
        # Directory prediction
        if not os.path.isdir(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return

        # For directory, we'll check each image individually
        # but print a summary about auto-resize behavior
        if args.force_original_size:
            print("\nProcessing all images at original size")
        else:
            print(
                f"\nAuto-resize enabled: images far from {args.target_size}x{args.target_size} will be resized"
            )

        predict_image_directory(inference_engine, args.image_dir, args)


if __name__ == "__main__":
    main()
