# MNIST Digit Recognition with PyTorch

A PyTorch implementation for recognizing handwritten digits from the MNIST dataset. This project uses a Convolutional Neural Network (CNN) with adaptive pooling, allowing it to handle images of various resolutions, not just the standard 28x28.

## How it Works

The model, `MNISTConvNet`, uses three convolutional blocks with proper batch normalization followed by an `AdaptiveAvgPool2d` layer. This adaptive pooling is the key feature, as it allows the model to accept images of any size during inference, even though it's trained on 28x28 images. This makes the model flexible for real-world scenarios where image sizes can vary.

**Key Features:**
- 3 convolutional blocks with batch normalization for each layer
- Adaptive pooling enables multi-resolution inference
- ~616K parameters, ~99.4-99.6% accuracy on MNIST
- Train on 28x28, infer on any size (64x64, 224x224, 512x512, etc.)

## Dependencies

Project dependencies are managed in the `pyproject.toml` file.

- Python 3.13+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- pillow
- tqdm

## How to Run

1.  **Install dependencies:**

    ```bash
    # Using uv (recommended)
    uv sync
    ```

2.  **Configuration:**

    Before running training or evaluation, create a `config.toml` file by copying the example:

    ```bash
    cp configs/config_example.toml configs/config.toml
    ```

    Then, you can modify `configs/config.toml` to change hyperparameters, paths, and other settings.

3.  **Quick Demo:**

    Run the demo to train the model and see multi-resolution inference in action:

    ```bash
    uv run python scripts/demo.py
    ```

4.  **Training:**

    To run a training session with the default settings from `config.toml`:

    ```bash
    uv run python scripts/train.py
    ```

    You can also specify a preset:

    ```bash
    uv run python scripts/train.py --preset high_accuracy
    ```

    Command-line arguments will override the settings in `config.toml`:

    ```bash
    uv run python scripts/train.py --preset high_accuracy --epochs 100
    ```

5.  **Inference:**

    To perform inference on a single image (multi-resolution enabled by default):

    ```bash
    uv run python scripts/inference.py \
        --checkpoint checkpoints/best_model.pth \
        --image path/to/your/digit.png
    ```

    The model automatically handles any input size without resizing!

6.  **Evaluation:**

    Evaluate a trained model on the test set:

    ```bash
    uv run python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pth
    ```

## Project Structure

```
mnist-digit-recognition/
├── data/                   # MNIST data (ignored by git)
├── checkpoints/            # Model checkpoints (ignored by git)
├── logs/                   # Logs and visualizations (ignored by git)
├── examples/
│   └── multi_resolution_example.py  # Example code for multi-resolution inference
├── scripts/                # Main scripts
│   ├── demo.py             # Complete demo with training and multi-resolution inference
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluate a trained model
│   ├── inference.py        # Run inference on images
│   └── detect_hardware.py  # Utility to detect hardware
├── src/
│   ├── models/
│   │   └── cnn.py          # CNN model architectures
│   └── utils/
│       ├── data_loader.py  # Data loading utilities
│       ├── trainer.py      # Training loop logic
│       ├── inference.py    # Inference helper functions
│       ├── visualize.py    # Visualization utilities
│       └── config_loader.py # Loads config from TOML
│       └── __init__.py
├── configs/
│   ├── config_example.toml   # Example configuration file
│   └── config.toml           # Local configuration (ignored by git)
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```
