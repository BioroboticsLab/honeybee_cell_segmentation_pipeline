
# Mask Writer

Convert JSON annotations to PNG segmentation masks.


## Installation

```bash
# From repository root
python install.py mask-writer
```

Or with Poetry (if you prefer):

```bash
poetry install
poetry run mask-writer <input_path> [...]
```

This tool is independent and has no internal dependencies.


## Usage

You can run `mask-writer` from the project root, the mask_writer directory, or anywhere with the environment active:


```sh
mask-writer <input_path> [--input_json_path <json_path>] [--label_classes_path <label_path>] [--masks_output_path <output_path>] [--visualize_indices N [N ...]]
```

- `<input_path>`: Path to the directory containing your input images (`.png` files).
- `--input_json_path <json_path>`: (Optional) Path to the directory containing the annotation JSON files. Defaults to `<input_path>`.
- `--label_classes_path <label_path>`: (Optional) Path to the directory containing `label_classes.json`. Defaults to `<input_path>`.
- `--masks_output_path <output_path>`: (Optional) Path to the directory where the output masks will be saved. Defaults to `<input_path>`.
- `--visualize_indices N [N ...]`: (Optional) List of integer indices (space separated) specifying which images' masks to visualize after creation. For example, `--visualize_indices 0 2 5` will show the masks for the 1st, 3rd, and 6th images.

## Example


```sh
mask-writer ./images --input_json_path ./annotations --label_classes_path ./labels --masks_output_path ./masks --visualize_indices 0 2
```

This will:
1. Read JSON annotation files
2. Convert each to a PNG mask
3. Save to the specified output directory

#### Output

- For each image, a mask PNG is created in the `ground_truth_masks` subdirectory of your output path.
- Each pixel in the mask is assigned an integer value according to its label:
	- `0`: background/unlabeled
	- Other values: as defined in your label config JSON
