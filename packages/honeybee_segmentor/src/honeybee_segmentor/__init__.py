"""Honeybee segmentation framework for comb analysis."""

__version__ = "0.1.0"

# Re-export main classes for easier imports
try:
    from .model.HoneyBeeCombSegmentationModel import HoneyBeeCombSegmentationModel
    from .inference.HoneyBeeCombInferer import HoneyBeeCombInferer
    from .dataset.CustomDataset import CustomDataset

    __all__ = [
        "HoneyBeeCombSegmentationModel",
        "HoneyBeeCombInferer",
        "CustomDataset",
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed yet
    pass
