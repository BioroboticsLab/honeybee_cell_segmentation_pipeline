
# Frame Extractor

Extract frames from video files at regular intervals using FFmpeg.

## Installation & Setup

1. Install the package (from project root):
   ```bash
   python install.py frame-extractor
   ```

2. Provide ffmpeg. The extractor resolves ffmpeg in this order: explicit
   `--ffmpeg-bin` / `ffmpeg_bin_path` argument → `BB_FFMPEG_BIN` env var →
   a binary bundled under `heavy_preprocessing/frame_extractor/src/frame_extractor/bin/`
   → `ffmpeg` found on `PATH` (e.g. a conda env or container).

   For NVIDIA GPU decode (`hevc_cuvid`, the default), ffmpeg must be a CUDA build:
   - **Windows:** [Gyan FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/#git-master-builds)
   - **Linux:** [BtbN FFmpeg builds](https://github.com/BtbN/FFmpeg-Builds/releases)

   On CPU-only nodes (or with a plain ffmpeg that lacks NVDEC), pass
   `--decoder none` (CLI) or `decoder=None` (library) to use software decode.

## Input Directory Structure

Your input directory should look like this (this reflects the file structure of the BeesBook capturing setup):

```
input_dir/
├── 20250603/
│   ├── cam-0/
│   │   ├── cam-0_20250603T120000.000000.000Z--20250603T121000.000000.000Z.mp4
│   │   ├── cam-0_20250603T120000.000000.000Z--20250603T121000.000000.000Z.txt
│   │   └── ...
│   ├── cam-1/
│   │   └── ...
│   └── ...
├── 20250604/
│   └── ...
└── ...
```

- Each day is a folder named `YYYYMMDD`
- Each camera is a folder named `cam-N`
- Each camera folder contains `.mp4` video files and matching `.txt` timestamp files

## Usage

### Command Line

```bash
frame-extractor <base_dir> <out_dir> [--file_format png] [--interval_in_sec 5] [--max_workers 2] [--fps 3]
```

**Arguments:**
- `<base_dir>`: Path to the input directory (see structure above)
- `<out_dir>`: Path to the output directory for extracted frames
- `--file_format`: Output image format (default: png)
- `--interval_in_sec`: Interval in seconds between frames (default: 5)
- `--max_workers`: Number of parallel workers (default: 2)
- `--fps`: FPS rate of the videos (default: 3)

### Example

```bash
frame-extractor ./input_dir ./output_dir --interval_in_sec 60 --max_workers 4 --fps 3
```

### As a Library

```python
from pathlib import Path
from frame_extractor.global_video_processor import GlobalVideoProcessor

processor = GlobalVideoProcessor(
    base_dir=Path("./input_dir"),
    out_dir=Path("./output_dir"),
    interval_in_sec=60,
    max_workers=2,
    fps=3,
    dates=["20250603"],   # optional: restrict to specific YYYYMMDD folders
    cams=["cam-0"],        # optional: restrict to specific cameras
    decoder="hevc_cuvid",  # or None for software decode
)
processor.run()
```

Re-running at a coarser interval that is a multiple of a finer one does ~no
work: the extractor skips any video whose expected output frames already exist
(the coarser run's filenames are a subset of the finer run's).
