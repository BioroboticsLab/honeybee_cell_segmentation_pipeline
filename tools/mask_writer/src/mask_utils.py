from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def visualize_mask_by_index(output_path: Path, index: int, label_data: list[dict]) -> None:
    files = sorted(output_path.glob("*.png"))
    if index < 0 or index >= len(files):
        raise IndexError(f"Index {index} is out of range. Found {len(files)} mask files.")

    img_path = files[index]
    mask = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    indices = [0] + [entry["png_index"] for entry in label_data]
    names = ["unlabeled"] + [entry["name"] for entry in label_data]
    colors = ["#000000"] + [entry["color"] for entry in label_data]  # Black for unlabeled

    rgb_colors = [hex_to_rgb(c) for c in colors]

    max_index = max(indices)

    color_list = [rgb_colors[0]] * (max_index + 1)
    for idx, col in zip(indices, rgb_colors):
        color_list[idx] = col
    cmap = ListedColormap(color_list)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=max_index)
    plt.title(img_path.name)
    plt.axis("off")

    legend_elements = [Patch(facecolor=rgb_colors[i], edgecolor="k", label=names[i]) for i in range(len(names))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    plt.show()


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 8:
        r, g, b, a = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))
        return (r / 255, g / 255, b / 255, a / 255)
    elif len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (r / 255, g / 255, b / 255, 1.0)
    else:
        return (0, 0, 0, 1)
