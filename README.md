# Honeybee Cell Segmentation Pipeline

A modular monorepo for honeybee comb analysis, containing independent tools and ML packages for segmentation, annotation, and processing.

## Quick Start

### Installation

All tools can be installed using the unified installer script:

```bash
# List available tools
python install.py --list

# Install a specific tool (e.g., cell-finder)
python install.py cell-finder

# Install all tools (beware, this is huge)
python install.py --all
```

After installing tools that use deep learning, you'll see instructions for optional GPU acceleration.

### Installing Individual Packages (recommended)

You can also install packages directly with automatic dependency resolution:

```bash
# Install just the segmentation framework
python install.py honeybee-segmentor

# Install comb-limitor (automatically installs honeybee-segmentor)
python install.py comb-limitor
```

## Repository Structure

```
├── heavy_preprocessing/        # Heavy, do-once-first batch steps (run on HPC)
│   ├── frame_extractor/        # Video frame extraction
│   └── background_generator/   # Background image generation
│
├── tools/                      # Lighter, interactive tools (run together, later)
│   ├── annotation_tool/        # Napari-based annotation UI
│   ├── mask_writer/            # Mask generation from annotations
│   └── cell_finder/            # Cell detection and analysis
│
├── packages/                   # Shared packages
│   ├── honeybee_segmentor/     # Core segmentation framework
│   └── comb_limitor/           # Binary comb mask generation (dependency)
│
└── install.py                  # Unified installer script
```

> **`heavy_preprocessing/` vs `tools/`:** frame extraction and background
> generation are the long-running steps (1–2 weeks on a single machine) that
> must be run **first**, before anything else — they are kept separate to make
> that clear, and are intended to be scaled across an HPC cluster via `bb_hpc`
> (the `frame_extract` / `background` stages). The lighter `tools/` (cell
> finder, mask writer, annotation UI) run together afterwards.

## Detailed Documentation

Each tool and package has its own README with detailed usage instructions:

### Heavy preprocessing (run first, on HPC)

- **Frame Extractor**: `/heavy_preprocessing/frame_extractor/README.md`
- **Background Generator**: `/heavy_preprocessing/background_generator/README.md`

### Tools

- **Annotation Tool**: `/tools/annotation_tool/README.md`
- **Cell Finder**: `/tools/cell_finder/README.md`
- **Mask Writer**: `/tools/mask_writer/README.md`

### Packages

- **Honeybee Segmentor**: `/packages/honeybee_segmentor/README.md`
- **Comb Limitor**: `/packages/comb_limitor/README.md`
