# Honeybee Comb Limitor

A package for generating binary comb masks from honeybee images using deep learning segmentation.

## Description

This package uses the honeybee-segmentor framework with custom model weights to generate binary masks of the honeycomb region in images.

## Installation

### Automatic

```bash
# Install comb-limitor standalone (automatically installs honeybee_segmentor first)
python install.py comb-limitor

# Or install as part of cell-finder (installs: honeybee_segmentor → comb_limitor → cell_finder)
python install.py cell-finder
```

### Manual installation

If you prefer manual pip installation:

```bash
# From repository root - install dependencies in order
pip install -e ./packages/honeybee_segmentor
pip install -e ./packages/comb_limitor
```

## Usage


### Parameters

- `model_name`: Name of the segmentation model
- `model_dir`: Directory containing model weights
- `label_config`: Path to label configuration JSON
- `device`: "cuda" or "cpu"
- `apply_closing`: Apply morphological closing to mask
- `apply_outlier_suppression`: Remove small outlier regions
- `closing_kernel_size`: Size of morphological kernel
- `min_contour_area`: Minimum area for contours to keep

## Output

Returns a binary numpy array where:
- 1 = comb region
- 0 = background
