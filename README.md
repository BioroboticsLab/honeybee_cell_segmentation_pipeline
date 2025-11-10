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
├── tools/                      # Independent tools
│   ├── annotation_tool/        # Napari-based annotation UI
│   ├── mask_writer/            # Mask generation from annotations
│   ├── frame_extractor/        # Video frame extraction
│   ├── cell_finder/            # Cell detection and analysis
│   └── background_generator/   # Background image generation
│
├── packages/                   # Shared packages
│   ├── honeybee_segmentor/     # Core segmentation framework
│   └── comb_limitor/           # Binary comb mask generation (dependency)
│
└── install.py                  # Unified installer script
```

## Detailed Documentation

Each tool and package has its own README with detailed usage instructions:

### Tools

- **Annotation Tool**: `/tools/annotation_tool/README.md`
- **Cell Finder**: `/tools/cell_finder/README.md`
- **Frame Extractor**: `/tools/frame_extractor/README.md`
- **Mask Writer**: `/tools/mask_writer/README.md`
- **Background Generator**: `/tools/background_generator/README.md`

### Packages

- **Honeybee Segmentor**: `/packages/honeybee_segmentor/README.md`
- **Comb Limitor**: `/packages/comb_limitor/README.md`
