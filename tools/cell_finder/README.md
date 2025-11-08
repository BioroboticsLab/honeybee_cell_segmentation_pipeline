
# Cell Finder

Find and detect cells in honeybee comb images using template matching and graph building.

## Installation

```bash
# From repository root
python install.py cell-finder
```

This automatically installs:
- honeybee-segmentor (dependency)
- comb-limitor (dependency)
- cell-finder

## Usage

Run the Cell Finder to find as many cells as possible beforehand.
This will create a JSON file with the found cells, with the same name as the image.
Take `graph_building` as method, since the other methods don't perform well.

```sh
cell-finder <input_path> [--output_path <output_path>] <method> [--curve_aware]
```

- `<input_path>`: Path to the directory containing input images (`.png` files).
- `--output_path`: (Optional) Path to save results. Defaults to the input path.
- `<method>`: Detection method to use. Options:
	- `template_matching`
	- `circle_hough_transform`
	- `hybrid`
	- `graph_building` (recommended)
- `--curve_aware`: (Optional, only for `graph_building`) Enables curve-aware lattice estimation for curved honeycomb patterns. Needs to be the last flag.

## Example

```sh
cell-finder ./images graph_building --curve_aware
```

This will:
1. Use the comb limitor to find the comb region
2. Detect cells using template matching
3. Fill in missing cells using graph building
4. Save results as JSON files

---

## Performance Validation

You can validate the detection results against ground truth annotations using the performance validation tool.
This helps tune parameters and objectively assess detection quality, especially if you need to decide if `curve_aware` or `lattice_vector` works better.

```sh
performance-validation <predictions_path> [--ground_truth_path <ground_truth_path>] [--visualize]
```
- `<predictions_path>`: Required. Path to the directory containing predicted cell JSON files (and images).
- `--ground_truth_path <ground_truth_path>`: Optional. Path to the directory containing ground truth JSON files and images. If omitted, defaults to `performance_validation/ground_truth` (relative to the script).
- `--visualize`: Optional. If set, generates an HTML report with visualizations of the results.

- Compares detected cells to ground truth JSON files.
- Reports precision, recall, and F1 score.
- Visualizes true positives, false positives, and false negatives.
- There are a few annotated ground truth images in `cell_finder/performance_validation/ground_truth`
