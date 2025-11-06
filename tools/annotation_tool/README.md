
# Annotation Tool

Interactive tool for labeling honeybee comb cells in images using Napari. The tool uses Napari as framework (https://napari.org/stable/)

## Features
- Visualize and annotate cell positions and labels on honeybee comb images
- Add, move, resize, and delete cell points
- Assign labels from a customizable label set
- Save annotations as JSON files for each image

## Installation

Install via the project installer script (recommended):

```sh
python install.py annotation-tool
```

Or with Poetry (if you prefer):

```sh
poetry install
poetry run annotator <data_dir>
```

## Usage

Run the annotation tool from the command line:

```sh
annotator <data_dir> [--config <config_path>]
```

- `<data_dir>`: Path to the directory containing your input images (`.png` files) and annotation JSON files.
- `--config <config_path>`: (Optional) Path to a custom label config file (JSON). If not provided, uses the default `data/label_classes.json`.

When you start annotating, if an image does not have a corresponding labels JSON, an empty one will be created automatically.

### Interface Overview

- **Image viewer**: Displays the current image
- **Points layer**: Each point marks a cell to annotate
- **Brush layer**: Select multiple points at once
- **Label menu**: Choose the label to assign to selected points
- **Navigation buttons**: Move between images
- **Reset button**: Undo the last change
- **Label legend**: Shows all available labels and their colors

### Layers

#### Points Layer
- Add points: Click on the image
- Select points: Click, use selection rectangle, or brush tool
- Move points: Drag selected points
- Resize points: Hold Alt and scroll mouse wheel
- Delete points: Select and press Delete

#### Brush Layer
- Activate: Press `B` or select in the layer list
- Select points: Paint over the area
- Adjust brush size: Use the brush size slider
- Note: Deleting points does not work when brush layer is activated. Switch to points layer first

### Key Bindings

| Key Combination | Action |
|-----------------|--------|
| `Ctrl+L`        | Toggle lock labeled points mode. Prevents selecting already labeled points (only 'unlabeled' can be selected now) |
| `J`             | Apply selected label to selected points |
| `B`             | Activate brush layer |
| `C`             | Activate points layer |
| `Y`             | Hide points (opacity = 0) |
| `X`             | Show points (opacity = 1) |
| `Ctrl+I`        | Delete all points except selected ones (Inverse deletion). To remove cells outside of the frame |
| `Delete`        | Delete selected points |

### Output

Annotations are saved as JSON files with the same base name as the image. Each file contains an array of cell objects:

```json
[
    {
        "id": "fc4e04d4-7669-4321-8516-0f37ac1b47e3",
        "center_x": 2176.0,
        "center_y": 2965.0,
        "radius": 24.0,
        "label": "nectar"
    },
    ...
]
```