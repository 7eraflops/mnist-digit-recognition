#!/usr/bin/env python3
"""
Hardware detection script for MNIST Digit Recognition project.
Detects available GPU hardware and recommends the correct uv sync command.
"""

import subprocess
import sys
import os


def check_nvidia_gpu():
    """Check for NVIDIA GPU and CUDA version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]

            # Try to get CUDA version
            cuda_result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            cuda_version = None
            if cuda_result.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                for line in cuda_result.stdout.split("\n"):
                    if "CUDA Version:" in line:
                        cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                        break

            return True, gpu_name, cuda_version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None, None


def check_amd_gpu():
    """Check for AMD GPU and ROCm version."""
    try:
        # Check rocm-smi
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = None
            for line in result.stdout.split("\n"):
                if "GPU[0]" in line and "Card Series:" in line:
                    # Extract GPU name after "Card Series:"
                    parts = line.split("Card Series:")
                    if len(parts) > 1:
                        gpu_name = parts[1].strip()
                        break

            # Try to get ROCm version
            rocm_version = None
            version_paths = ["/opt/rocm/.info/version", "/opt/rocm/VERSION"]
            for version_path in version_paths:
                if os.path.exists(version_path):
                    try:
                        with open(version_path, "r") as f:
                            rocm_version = f.read().strip()
                            break
                    except:
                        pass

            return True, gpu_name or "AMD GPU", rocm_version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None, None


def recommend_command(has_nvidia, nvidia_info, has_amd, amd_info):
    """Recommend the appropriate uv sync command."""
    if has_nvidia:
        gpu_name, cuda_version = nvidia_info
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        if cuda_version:
            print(f"  CUDA Version: {cuda_version}")
            major_minor = ".".join(cuda_version.split(".")[:2])

            # Recommend based on CUDA version
            if cuda_version.startswith("13"):
                print("\n📦 Recommended installation command:")
                print("  uv sync --index https://download.pytorch.org/whl/cu130")
            elif cuda_version.startswith("12.8"):
                print("\n📦 Recommended installation command:")
                print("  uv sync --index https://download.pytorch.org/whl/cu128")
            elif cuda_version.startswith("12.6"):
                print("\n📦 Recommended installation command:")
                print("  uv sync --index https://download.pytorch.org/whl/cu126")
            else:
                print(
                    "\n📦 Recommended installation command (using CUDA 12.8, most compatible):"
                )
                print("  uv sync --index https://download.pytorch.org/whl/cu128")
        else:
            print(
                "\n📦 Recommended installation command (using CUDA 12.8, most compatible):"
            )
            print("  uv sync --index https://download.pytorch.org/whl/cu128")

        print("\n💡 Alternative CUDA versions:")
        print("  CUDA 13.0: uv sync --index https://download.pytorch.org/whl/cu130")
        print("  CUDA 12.8: uv sync --index https://download.pytorch.org/whl/cu128")
        print("  CUDA 12.6: uv sync --index https://download.pytorch.org/whl/cu126")
        return True

    if has_amd:
        gpu_name, rocm_version = amd_info
        print(f"✓ AMD GPU detected: {gpu_name}")
        if rocm_version:
            print(f"  ROCm Version: {rocm_version}")

            # Recommend based on ROCm version
            if rocm_version.startswith("7"):
                print("\n📦 Recommended installation command:")
                print("  uv sync --extra rocm")
            elif rocm_version.startswith("6.4"):
                print("\n📦 Recommended installation command:")
                print(
                    "  uv sync --extra rocm --index https://download.pytorch.org/whl/rocm6.4"
                )
            else:
                print("\n📦 Recommended installation command (using ROCm 7.0):")
                print("  uv sync --extra rocm")
        else:
            print("\n📦 Recommended installation command (using ROCm 7.0):")
            print("  uv sync --extra rocm")

        print("\n💡 Alternative ROCm versions:")
        print("  ROCm 7.0: uv sync --extra rocm")
        print(
            "  ROCm 6.4: uv sync --extra rocm --index https://download.pytorch.org/whl/rocm6.4"
        )
        return True

    return False


def main():
    print("🔍 Detecting hardware...\n")

    has_nvidia, nvidia_name, cuda_version = check_nvidia_gpu()
    has_amd, amd_name, rocm_version = check_amd_gpu()

    if recommend_command(
        has_nvidia, (nvidia_name, cuda_version), has_amd, (amd_name, rocm_version)
    ):
        print("\n✨ After installation, verify with:")
        print(
            "  uv run python -c \"import torch; print('GPU:', torch.cuda.is_available())\""
        )
    else:
        print("⚠ No GPU detected or GPU drivers not installed.")
        print("\n📦 Recommended installation command (CPU-only):")
        print("  uv sync --index https://download.pytorch.org/whl/cpu")
        print("\n💡 If you have a GPU:")
        print("  - NVIDIA: Install CUDA drivers and nvidia-smi")
        print("  - AMD: Install ROCm drivers and rocm-smi")
        print("  - Then run this script again")


if __name__ == "__main__":
    main()
