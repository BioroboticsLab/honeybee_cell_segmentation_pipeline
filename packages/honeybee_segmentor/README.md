# Honeybee Segmentor

Core segmentation framework for honeybee comb analysis. This is basically the original framework by Ivan Matoschchuk with a few modifications.

## Description

This package provides PyTorch-based segmentation models and inference pipelines for analyzing honeybee comb images. It's used as a dependency by other tools in the pipeline. Currently the background_generator and the comb_limitor.

## Installation

### Standard Installation (CPU-only)

```bash
# From the repository root
pip install -e ./packages/honeybee_segmentor[dev]
```

### Installation with GPU Support (Recommended)

For GPU acceleration with CUDA, you need to install PyTorch with CUDA support **after** installing the package:

```bash
# Step 1: Install the package (this installs CPU-only PyTorch)
pip install -e ./packages/honeybee_segmentor[dev]

# Step 2: Reinstall PyTorch with CUDA support
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Why this order?**
- Installing PyTorch with CUDA *first* can cause dependency conflicts with packages like monai and segmentation-models-pytorch
- Installing it *after* ensures all dependencies are satisfied, then replaces PyTorch with the CUDA version

**Verify CUDA installation:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")  # Should show: 2.7.1+cu118
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be: True
```

### CUDA Version Selection

- **CUDA 11.8**: `--index-url https://download.pytorch.org/whl/cu118` (recommended)
- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`

Check your NVIDIA driver's maximum CUDA version:
```bash
nvidia-smi
```

## Usage

```python
from honeybee_segmentor.model import HoneyBeeCombSegmentationModel
from honeybee_segmentor.inference import HoneyBeeCombInferer

# Load model
model = HoneyBeeCombSegmentationModel(
    model_name="unet_effnetb0",
    path_to_pretrained_models="path/to/models",
    device="cuda"
)

# Create inferer
inferer = HoneyBeeCombInferer(
    model_name="unet_effnetb0",
    path_to_pretrained_models="path/to/models",
    label_classes_config="path/to/label_classes.json",
    device="cuda"
)

# Run inference
mask = inferer.infer(image_path)
```

## Model Files

Pre-trained model weights should be placed in the `models/` directory at the package root.

## Configuration

Configuration files for inference parameters can be found in `config/config.yaml`.
