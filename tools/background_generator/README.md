# Background Image Generator

Generate bee-free background images from video by segmenting and removing bees from frames, then computing rolling median backgrounds.

## Installation

Install using the unified installer from the repository root:

```bash
python install.py background-generator
```

This will automatically install both `background-generator` and its dependency `frame-extractor`.

### Poetry Installation

If you use Poetry, you can install the background generator and its dependencies with:

```bash
poetry install
```

To activate the Poetry environment:

```bash
poetry shell
```

You can then run the CLI as usual:

```bash
background-generator --help
```

### Manual Installation

If you prefer manual installation:

```bash
cd tools/background_generator
pip install -e .
```

**Note:** You may need to reinstall PyTorch with CUDA support after installation (you may need another version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

The background generator can be used as a **command-line tool** or imported as a **Python module**.

### Command-Line Interface

After installation, use the `background-generator` command:

#### Basic Usage (frames already extracted)

```bash
background-generator --source /path/to/extracted/frames --output /path/to/backgrounds
```

#### Full Pipeline (extract frames + generate backgrounds)

```bash
background-generator \
  --extract-frames \
  --video-dir /path/to/videos \
  --source /path/to/extracted/frames \
  --output /path/to/backgrounds \
  --interval 60 \
  --max-workers 2
```

#### Advanced Configuration

```bash
background-generator \
  --source /path/to/frames \
  --output /path/to/backgrounds \
  --window-size 10 \
  --num-median-images 200 \
  --max-cycles 10 \
  --apply-clahe post \
  --mask-dilation 15 \
  --median-computation cupy \
  --device cuda
```

### CLI Options

**Required:**
- `--source`: Directory containing extracted frames (organized by camera, e.g., `cam-0/`, `cam-1/`)
- `--output`: Output directory for background images

**Frame Extraction (optional):**
- `--extract-frames`: Extract frames from video before generating backgrounds
- `--video-dir`: Directory containing video files (required if `--extract-frames` is set)
- `--interval`: Frame extraction interval in seconds (default: 60)
- `--max-workers`: Parallel workers for extraction (default: 2)
- `--fps`: Video processing FPS (default: 3)

**Background Generation:**
- `--window-size`: Frames per rolling median (default: 10)
- `--num-median-images`: Number of median images per background (default: 200)
- `--max-cycles`: Max backgrounds per camera (default: None = unlimited)
- `--jump-size`: Step size between backgrounds (default: 1)
- `--apply-clahe`: CLAHE timing: `intermediate` or `post` (default: post)
- `--mask-dilation`: Dilation kernel size: 0, 9, 15, or 25 (default: 15)
- `--median-computation`: Method: `cupy` (GPU), `cuda_support`, or `masked_array` (CPU) (default: cupy)
- `--device`: Processing device: `cuda` or `cpu` (default: cuda)

### Python Module Usage

```python
from pathlib import Path
from background_img_generator import BackgroundImageGenerator
from utils import BgImageGenConfig

# Configure background generation
config = BgImageGenConfig(
    window_size=10,
    num_median_images=200,
    max_cycles=10,
    apply_clahe="post",
    mask_dilation=15,
    median_computation="cupy",
    device="cuda"
)

# Generate backgrounds
generator = BackgroundImageGenerator(
    source_path=Path("/path/to/frames"),
    output_path=Path("/path/to/backgrounds"),
    config=config
)
generator.run()
```

### Optional: Extract Frames First

```python
from pathlib import Path
from global_video_processor import GlobalVideoProcessor

processor = GlobalVideoProcessor(
    base_dir=Path("/path/to/videos"),
    out_dir=Path("/path/to/frames"),
    interval_in_sec=60,
    max_workers=2,
    fps=3
)
processor.run()
```

## Configuration Details

See `BgImageGenConfig` docstring in `utils.py` for detailed parameter explanations.

## Notes

- The generator automatically resumes from the last processed image
- Bee segmentation uses the honeybee segmentation model (classes 1 and 8 are masked)
- Runs incrementally - you can stop and restart without losing progress
