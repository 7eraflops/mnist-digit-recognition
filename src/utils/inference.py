"""
Inference utilities for MNIST digit recognition.

This module provides functions for making predictions on new images,
including preprocessing, single image inference, and batch inference.
Supports both fixed-size models (28x28) and adaptive models (any resolution).
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class MNISTInference:
    """
    Inference class for MNIST digit recognition.

    This class handles loading trained models and making predictions on new images.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        adaptive: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            model: PyTorch model for inference
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to run inference on (CPU/GPU)
            adaptive: Whether the model can handle variable input sizes (default: False)
            target_size: Target size for preprocessing. If None, uses (28, 28) for
                        non-adaptive models or preserves original size for adaptive models
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        self.adaptive = adaptive
        self.target_size = target_size if target_size is not None else (28, 28)

        # Define preprocessing transform
        if adaptive and target_size is None:
            # For adaptive models without specified target size, don't resize
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            # For fixed-size models or when target size is specified
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from {checkpoint_path}")

    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        preserve_size: bool = None,
        invert: bool = True,
        center_digit: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            preserve_size: Override for whether to preserve original size.
                          If None, uses self.adaptive setting
            invert: If True, inverts the image (for black-on-white drawings).
                   MNIST expects white digits on black background.
            center_digit: If True, centers the digit with proper padding like MNIST

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if preserve_size is None:
            preserve_size = self.adaptive and self.target_size == (28, 28)
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image).convert("L")  # Ensure grayscale

        # Convert numpy array to PIL Image
        elif isinstance(image, np.ndarray):
            # Handle different array shapes
            if image.ndim == 3:
                if image.shape[0] in [1, 3]:  # Channel first
                    image = image.transpose(1, 2, 0)
                if image.shape[2] == 1:  # Single channel
                    image = image.squeeze()

            # Normalize if needed (assume 0-255 range)
            if image.max() > 1.0:
                image = image.astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)

            image = Image.fromarray(image)

        # Apply transformations
        if preserve_size and self.adaptive:
            # Use transform without resize for adaptive models
            transform_no_resize = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            image_tensor = transform_no_resize(image)
        else:
            image_tensor = self.transform(image)

        return image_tensor

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        return_probabilities: bool = False,
    ) -> Union[int, Tuple[int, torch.Tensor]]:
        """
        Predict digit class for a single image.

        Args:
            image: Input image (various formats supported)
            return_probabilities: Whether to return class probabilities

        Returns:
            int or Tuple[int, torch.Tensor]: Predicted class and optionally probabilities
        """
        # Preprocess image if not already a tensor
        if not isinstance(image, torch.Tensor):
            image_tensor = self.preprocess_image(image)
        else:
            image_tensor = image

        # Add batch dimension if needed
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        if return_probabilities:
            return predicted_class, probabilities.squeeze().cpu()
        else:
            return predicted_class

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 32,
        return_probabilities: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Predict digit classes for a batch of images.

        Args:
            images: List of input images
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities

        Returns:
            List[int] or Tuple[List[int], torch.Tensor]: Predicted classes and optionally probabilities
        """
        all_predictions = []
        all_probabilities = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                img_tensor = self.preprocess_image(img)
                batch_tensors.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Make predictions
            with torch.no_grad():
                output = self.model(batch_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predicted_classes.cpu().numpy().tolist())
            all_probabilities.append(probabilities.cpu())

        if return_probabilities:
            all_probabilities = torch.cat(all_probabilities, dim=0)
            return all_predictions, all_probabilities
        else:
            return all_predictions

    def predict_with_confidence(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[int, float]:
        """
        Predict digit class with confidence score.

        Args:
            image: Input image

        Returns:
            Tuple[int, float]: Predicted class and confidence (0-100)
        """
        predicted_class, probabilities = self.predict(image, return_probabilities=True)
        confidence = probabilities[predicted_class].item() * 100

        return predicted_class, confidence

    def predict_top_k(
        self, image: Union[str, np.ndarray, Image.Image], k: int = 3
    ) -> Tuple[List[int], List[float]]:
        """
        Predict top-k digit classes with probabilities.

        Args:
            image: Input image
            k: Number of top predictions to return

        Returns:
            Tuple[List[int], List[float]]: Top-k classes and their probabilities
        """
        _, probabilities = self.predict(image, return_probabilities=True)

        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, k)

        top_k_classes = top_k_indices.numpy().tolist()
        top_k_probs = (top_k_probs.numpy() * 100).tolist()

        return top_k_classes, top_k_probs

    def visualize_prediction(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Visualize prediction with probabilities for all classes.

        Args:
            image: Input image
            save_path: Path to save the visualization (optional)
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt

        # Get prediction and probabilities
        predicted_class, probabilities = self.predict(image, return_probabilities=True)

        # Preprocess for display
        if isinstance(image, str):
            display_img = Image.open(image).convert("L")
        elif isinstance(image, torch.Tensor):
            # Handle torch tensors - convert to numpy and squeeze
            img_np = image.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                display_img = img_np.squeeze(0)
            elif img_np.ndim == 4:  # Batch dimension
                display_img = img_np.squeeze(0).squeeze(0)
            else:
                display_img = img_np
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 1:
                display_img = image.squeeze(0)
            else:
                display_img = image
        else:
            display_img = image

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Display image
        if isinstance(display_img, Image.Image):
            ax1.imshow(display_img, cmap="gray")
        else:
            ax1.imshow(display_img, cmap="gray")
        ax1.axis("off")
        ax1.set_title(
            f"Predicted: {predicted_class}\nConfidence: {probabilities[predicted_class] * 100:.2f}%",
            fontsize=14,
            fontweight="bold",
        )

        # Plot probabilities
        classes = list(range(10))
        probs = probabilities.numpy() * 100
        colors = ["green" if i == predicted_class else "blue" for i in classes]

        ax2.barh(classes, probs, color=colors, alpha=0.7)
        ax2.set_xlabel("Probability (%)", fontsize=12)
        ax2.set_ylabel("Digit Class", fontsize=12)
        ax2.set_title("Class Probabilities", fontsize=14, fontweight="bold")
        ax2.set_yticks(classes)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Prediction visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def load_model_for_inference(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    adaptive: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
) -> MNISTInference:
    """
    Convenience function to load a model for inference.

    Args:
        model: PyTorch model architecture
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        adaptive: Whether the model can handle variable input sizes
        target_size: Target size for preprocessing

    Returns:
        MNISTInference: Inference engine ready for predictions
    """
    return MNISTInference(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        adaptive=adaptive,
        target_size=target_size,
    )
