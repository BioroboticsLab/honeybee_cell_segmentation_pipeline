import cv2
import gc
import numpy as np
import tempfile

from joblib import Parallel, delayed
from pathlib import Path, PurePath
from scipy.stats import mode
from tqdm import tqdm
from typing import List, Literal
import honeybee_segmentor
from honeybee_segmentor.inference.HoneyBeeCombInferer import (
    HoneyBeeCombInferer,
)
from collections import deque
from typing import Deque
from background_generator.utils import timed, BgImageGenConfig

import torch
import cupy as cp
import re


class BackgroundImageGenerator:
    def __init__(self, source_path: Path, output_path: Path, config: BgImageGenConfig):
        self._config = config
        if not source_path.is_dir():
            raise NotADirectoryError(f"provided source path {source_path} is not a directory")
        self.source_path = source_path
        if not output_path.is_dir():
            raise NotADirectoryError(f"provided output path {output_path} is not a directory")
        self.output_path = output_path
        self.frame_dirs_per_cam = self._find_extracted_frames_dirs()

        self.output_dirs = self.create_output_dir()
        weights_path: Path = self._get_weiths_path()
        self.model = HoneyBeeCombInferer(
            model_name=self._config.segmentation_model,
            path_to_pretrained_models=str(weights_path),
            device=self._config.device,
        )

    def run(self) -> None:
        for cam, path in self.frame_dirs_per_cam.items():
            out_dirs_per_cam = self.output_dirs.get(cam)
            cam_masked_path = out_dirs_per_cam.get("masked")
            cam_bg_path = out_dirs_per_cam.get("background")
            self.mask_out_bees(cam_in_path=path, cam_masked_out_path=cam_masked_path)
            self.process_all_rolling_backgrounds(
                masked_img_dir=cam_masked_path,
                background_img_dir=cam_bg_path,
                jump_size_from_last=self._config.jump_size_from_last,
                max_cycles=self._config.max_cycles,
            )

    def _find_extracted_frames_dirs(self) -> dict[str, Path]:
        pattern = re.compile(r"^cam-\d$")
        matches = {}
        for path in self.source_path.iterdir():
            if path.iterdir() and pattern.match(path.name):
                matches[path.name] = path
        return matches

    def _find_images_by_path(
        self,
        path: Path,
        role: Literal["background", "masked"],
    ) -> list[Path]:

        patterns = [
            r"^{prefix}_cam-\d_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_cam-\d_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)--(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)--(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
        ]
        regexes = [re.compile(p.format(prefix=role)) for p in patterns]

        all_images = sorted(path.glob("*"))
        filtered_images = [img for img in all_images if any(r.match(img.name) for r in regexes)]

        if len(filtered_images) == 0:
            print(f"no images found that match any pattern. searched for role: {role}")

        return filtered_images

    def mask_out_bees(self, cam_in_path: Path, cam_masked_out_path: Path) -> None:

        image_files = self.find_unmasked_imgages(cam_in_path, cam_masked_out_path)

        for source_img_path in tqdm(image_files):
            img = cv2.imread(str(source_img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: could not read {source_img_path}")
                continue
            pred_mask = self.model.infer(img, return_logits=False)
            bee_pixels = (pred_mask == 1) | (pred_mask == 8)
            # img[bee_pixels] = 0
            refined_mask = self._refine_mask(bee_pixels)
            img[refined_mask > 0] = 0
            out_path = cam_masked_out_path / f"masked_{PurePath(source_img_path).name}"
            cv2.imwrite(str(out_path), img)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self._config.mask_dilation:
            return mask
        else:
            kernel_size = (self._config.mask_dilation, self._config.mask_dilation)
            kernel = np.ones(kernel_size, np.uint8)
            return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    def find_unmasked_imgages(self, cam_in_path: Path, masked_cam_out_path: Path) -> List[Path]:
        source_images = sorted(cam_in_path.glob("*.[pj][np][ge]*"))
        masked_images = set(f.name.replace("masked_", "") for f in masked_cam_out_path.glob("masked_*"))
        unmasked_images = [img for img in source_images if img.name not in masked_images]
        return unmasked_images

    def create_output_dir(self) -> dict[str, dict[str, Path]]:
        output_dir_dict = {}
        for key in self.frame_dirs_per_cam.keys():
            masked_img_dir: Path = self.output_path / "masked" / key
            Path.mkdir(masked_img_dir, parents=True, exist_ok=True)
            background_img_dir: Path = self.output_path / key
            Path.mkdir(background_img_dir, parents=True, exist_ok=True)
            output_dir_dict[key] = {"masked": masked_img_dir, "background": background_img_dir}
        return output_dir_dict

    def _read_image(self, filepath: Path) -> cv2.typing.MatLike:
        return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

    def _get_weiths_path(self) -> Path:
        segmentor_root = Path(honeybee_segmentor.__file__).parents[2]
        model_dir = segmentor_root / "models"
        assert model_dir.is_dir(), f"Weights path is not a dir: {model_dir}"
        return model_dir

    def process_all_rolling_backgrounds(
        self,
        masked_img_dir: Path,
        background_img_dir: Path,
        jump_size_from_last: int = 1,
        tile_size=(512, 512),
        use_median=True,
        max_cycles: int | None = None,
    ):
        cycle_count = 0
        while True:
            if max_cycles is not None and cycle_count >= max_cycles:
                print(f"Reached max_cycles limit ({max_cycles}). Stopping.")
                break

            image_created = self.process_rolling_backgrounds(
                masked_img_dir=masked_img_dir,
                background_img_dir=background_img_dir,
                jump_size_from_last=jump_size_from_last,
                tile_size=tile_size,
                use_median=use_median,
            )
            if not image_created:
                break

            cycle_count += 1

    @timed("Rolling Background Generation")
    def process_rolling_backgrounds(
        self,
        masked_img_dir: Path,
        background_img_dir: Path,
        jump_size_from_last: int,
        tile_size=(512, 512),
        use_median=True,
    ) -> bool:
        masked_images = self._find_images_by_path(masked_img_dir, role="masked")
        if not masked_images:
            print("No masked images found")
            print("masked dir", masked_img_dir)
            return False

        background_images = self._find_images_by_path(background_img_dir, role="background")

        image_queue: Deque[tuple[np.ndarray, Path]] = deque()

        last_processed_img_name = (
            background_images[-1].name.replace("background", "masked") if background_images else None
        )

        start_idx = 0
        if last_processed_img_name:
            try:
                last_index = masked_images.index(masked_img_dir / last_processed_img_name)
                start_idx = last_index + jump_size_from_last
            except ValueError:
                print(f"Could not find masked image {last_processed_img_name}")
                return False

        sampled_masked_paths = masked_images[start_idx::]

        if len(sampled_masked_paths) < self._config.window_size:
            print(f"Not enough images left for window. Found {len(sampled_masked_paths)}")
            return False

        total_possible = len(sampled_masked_paths) - self._config.window_size + 1
        if total_possible < self._config.num_median_images:
            print(
                f"Not enough images to compute {self._config.num_median_images} rolling medians. "
                f"Only {total_possible} possible. Skipping."
            )
            return False

        num_medians = self._config.num_median_images
        sampled_masked_paths = sampled_masked_paths[: num_medians + self._config.window_size - 1]

        print(f"Will compute {num_medians} rolling median frames")

        # First image for shape
        first_img = self._read_image(sampled_masked_paths[0])
        H, W = first_img.shape
        assert H > 0 and W > 0, f"Invalid shape: H={H}, W={W}"

        self._memmap_file = Path(tempfile.gettempdir()) / "rolling_medians.dat"
        self._rolling_memmap = np.memmap(self._memmap_file, dtype="uint8", mode="w+", shape=(num_medians, H, W))

        for path in sampled_masked_paths[: self._config.window_size - 1]:
            img = self._read_image(path)
            image_queue.append((img, path))

        median_index = 0
        paths_to_process = sampled_masked_paths[self._config.window_size - 1 :]

        for path in tqdm(paths_to_process, desc="Rolling median"):
            if median_index >= num_medians:
                break
            next_img = self._read_image(path)
            image_queue.append((next_img, path))
            if len(image_queue) == self._config.window_size:
                window_imgs = [img for img, _ in image_queue]
                match self._config.median_computation:
                    case "cuda_support":
                        background = self._compute_background_image_cuda_support(window_imgs, self._config.device)
                    case "cupy":
                        background = self._compute_background_image_cupy(window_imgs)
                    case "masked_array":
                        background = self._compute_background_image(window_imgs)
                if self._config.apply_clahe == "intermediate":
                    background = self._apply_clahe(background)
                self._rolling_memmap[median_index, :, :] = background
                median_index += 1
                image_queue.popleft()

        self._rolling_memmap.flush()
        print("Rolling medians written to disk.")

        # === Tile-wise global median ===
        print("Starting global background computation by tile...")

        del self._rolling_memmap
        gc.collect()

        self._rolling_memmap = np.memmap(self._memmap_file, dtype="uint8", mode="r", shape=(num_medians, H, W))

        background = np.zeros((H, W), dtype=np.uint8)
        results = Parallel(n_jobs=8)(
            delayed(self._process_tile_stack)(self._rolling_memmap, i, j, tile_size, use_median)
            for i in range(0, H, tile_size[0])
            for j in range(0, W, tile_size[1])
        )

        for i, i_end, j, j_end, tile_result in results:
            background[i:i_end, j:j_end] = tile_result

        if self._config.apply_clahe == "post":
            background = self._apply_clahe(background)

        bg_img_name = sampled_masked_paths[0].name.replace("masked", "background")

        self._save_image(background, background_img_dir / bg_img_name)
        print("Final background saved to:", background_img_dir / bg_img_name)

        self._cleanup_memory()

        return True

    # @timed("Tile Processing")
    def _process_tile_stack(
        self,
        stack: np.memmap,
        i: int,
        j: int,
        tile_size: tuple[int, int],
        use_median: bool,
    ) -> tuple[int, int, int, int, np.ndarray]:
        H, W = stack.shape[1:]  # (N, H, W)
        i_end = min(i + tile_size[0], H)
        j_end = min(j + tile_size[1], W)

        tile_stack = stack[:, i:i_end, j:j_end]
        N, th, tw = tile_stack.shape
        tile_flat = tile_stack.reshape(N, -1)
        out_tile = np.zeros((th * tw,), dtype=np.uint8)

        for k in range(th * tw):
            pixel_values = tile_flat[:, k]
            nonzero = pixel_values[pixel_values != 0]
            if len(nonzero) == 0:
                out_tile[k] = 0
            else:
                if use_median:
                    out_tile[k] = np.median(nonzero).astype(np.uint8)
                else:
                    val, _ = mode(nonzero, keepdims=True)
                    out_tile[k] = val[0].astype(np.uint8)

        return i, i_end, j, j_end, out_tile.reshape(th, tw)

    def _cleanup_memory(self):
        del self._rolling_memmap
        gc.collect()
        if self._memmap_file.exists():
            try:
                self._memmap_file.unlink()
                print(f"Deleted temporary memmap file {self._memmap_file}")
            except PermissionError as e:
                print(f"Could not delete memmap file: {e}")

    def _compute_background_image(self, images: list[np.ndarray]) -> np.ndarray:
        assert images, "No images provided."
        stacked = np.stack(images, axis=0)  # shape: (N, H, W)

        masked = np.ma.masked_equal(stacked, 0)
        median = np.ma.median(masked, axis=0).filled(0).astype(np.uint8)
        return median

    def _compute_background_image_cuda_support(self, images: list[np.ndarray], device: str = "cpu") -> np.ndarray:
        assert images, "No images provided."

        tensors = [torch.from_numpy(img).to(device=device, dtype=torch.float32) for img in images]
        stacked = torch.stack(tensors, dim=0)

        mask = stacked != 0
        stacked[~mask] = float("nan")

        median = torch.nanmedian(stacked, dim=0).values

        return median.nan_to_num(0).byte().cpu().numpy()

    def _compute_background_image_cupy(self, images: list[np.ndarray]) -> np.ndarray:
        assert images, "No images provided."

        stacked = cp.stack([cp.asarray(img, dtype=cp.uint8) for img in images], axis=0)
        stacked = stacked.astype(cp.float32)
        stacked[stacked == 0] = cp.nan

        median = cp.nanmedian(stacked, axis=0)

        result = cp.nan_to_num(median, nan=0).round().clip(0, 255).astype(cp.uint8)

        return cp.asnumpy(result)

    def _apply_clahe(self, img, clipLimit=2.0, tileGridSize=(8, 8)):

        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(img)

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        cv2.imwrite(str(output_path), image)
