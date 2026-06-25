import cv2
import gc
import os
import numpy as np
import tempfile

from datetime import datetime, timedelta
from joblib import Parallel, delayed
from pathlib import Path, PurePath
from scipy.stats import mode
from tqdm import tqdm
from typing import List, Literal
import honeybee_segmentor
from honeybee_segmentor.inference.HoneyBeeCombInferer import (
    HoneyBeeCombInferer,
)
from collections import deque, OrderedDict
from typing import Deque
from background_generator.utils import timed, BgImageGenConfig
from background_generator import windowing

import torch
import cupy as cp
import re


class BackgroundImageGenerator:
    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        config: BgImageGenConfig,
        cams: list[str] | None = None,
        dates: list[str] | None = None,
    ):
        self._config = config
        if not source_path.is_dir():
            raise NotADirectoryError(f"provided source path {source_path} is not a directory")
        self.source_path = source_path
        if not output_path.is_dir():
            raise NotADirectoryError(f"provided output path {output_path} is not a directory")
        self.output_path = output_path

        # Optional camera filter so an external scheduler (bb_hpc) can target one
        # (date, camera) shard per task. None -> all cameras under source_path.
        self._cams = set(cams) if cams else None

        # Optional calendar-day filter (YYYYMMDD strings) so a scheduler can shard
        # a flat frame folder by day -- one (cam, day) per task. None -> all days.
        # Only valid in windowed mode (enforced in run()); rolling/count-mode
        # resume is global and cannot be sharded by day.
        self._dates = set(dates) if dates else None

        # Per-task memmap location: defaults to system temp, but a per-task path
        # avoids collisions when several jobs share a node (the old code used a
        # single fixed /tmp/rolling_medians.dat).
        self._memmap_base = Path(config.memmap_dir) if config.memmap_dir else Path(tempfile.gettempdir())
        self._memmap_base.mkdir(parents=True, exist_ok=True)
        self._memmap_counter = 0

        self.frame_dirs_per_cam = self._find_extracted_frames_dirs()

        self.output_dirs = self.create_output_dir()
        weights_path: Path = self._get_weiths_path()
        self.model = HoneyBeeCombInferer(
            model_name=self._config.segmentation_model,
            path_to_pretrained_models=str(weights_path),
            device=self._config.device,
        )

    def run(self) -> None:
        if self._dates is not None and not self._config.background_window:
            raise ValueError(
                "dates filtering is only supported in windowed mode: set "
                "config.background_window (e.g. 'day'). Rolling/count-mode resume "
                "is global and cannot be sharded by day."
            )
        for cam, path in self.frame_dirs_per_cam.items():
            out_dirs_per_cam = self.output_dirs.get(cam)
            cam_masked_path = out_dirs_per_cam.get("masked")
            cam_bg_path = out_dirs_per_cam.get("background")
            self.mask_out_bees(cam_in_path=path, cam_masked_out_path=cam_masked_path)
            if self._config.background_window:
                # Time-windowed mode: one background per time window.
                self.process_windowed_backgrounds(
                    masked_img_dir=cam_masked_path,
                    background_img_dir=cam_bg_path,
                    min_frames=self._config.min_frames,
                )
            else:
                # Original count-driven rolling-median mode.
                self.process_all_rolling_backgrounds(
                    masked_img_dir=cam_masked_path,
                    background_img_dir=cam_bg_path,
                    jump_size_from_last=self._config.jump_size_from_last,
                    max_cycles=self._config.max_cycles,
                )

    # ------------------------------------------------------------------ #
    # Config-encoded output layout + temp paths
    # ------------------------------------------------------------------ #
    def config_tag(self) -> str:
        """A short tag identifying this background configuration.

        Encoded into the output path so different configs (e.g. 5-min vs 10-min
        sampling, hourly vs daily windows) are distinct, comparable products and
        a repeat config is recognized as already-done.
        """
        c = self._config
        return windowing.config_tag(
            c.frame_interval_sec, c.background_window, c.window_size, c.num_median_images
        )

    def _new_memmap_path(self, tag: str) -> Path:
        self._memmap_counter += 1
        return self._memmap_base / f"bgmedian_{os.getpid()}_{tag}_{self._memmap_counter}.dat"

    def _find_extracted_frames_dirs(self) -> dict[str, Path]:
        pattern = re.compile(r"^cam-\d$")
        matches = {}
        for path in self.source_path.iterdir():
            if path.is_dir() and pattern.match(path.name):
                if self._cams is not None and path.name not in self._cams:
                    continue
                matches[path.name] = path
        return matches

    def _find_images_by_path(
        self,
        path: Path,
        role: Literal["background", "masked"],
    ) -> list[Path]:
        """Find images by matching filename patterns for the given role.
        Falls back to a general '{prefix}_*.png' search if no pattern matches.
        """

        patterns = [
            r"^{prefix}_cam-\d_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_cam-\d_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)--(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
            r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)--(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{1,3}}Z)\.png$",
        ]
        regexes = [re.compile(p.format(prefix=role)) for p in patterns]

        all_images = sorted(path.glob("*"))
        filtered_images = [img for img in all_images if any(r.match(img.name) for r in regexes)]

        # If nothing matched, fall back to a simpler search
        if not filtered_images:
            print(f"No images found matching strict patterns for role '{role}'. Falling back to general search...")
            fallback_pattern = f"{role}_*.png"
            filtered_images = sorted(path.glob(fallback_pattern))

            if filtered_images:
                print(f"Using general search pattern '{fallback_pattern}', found {len(filtered_images)} images.")
            else:
                print(f"No images found for role '{role}' (in fallback search).")

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

    def _in_dates(self, path: Path) -> bool:
        """True if the frame's embedded timestamp falls on one of self._dates.

        Frames whose timestamp can't be parsed are excluded when a date filter is
        active (they can't be attributed to a calendar day).
        """
        ts = windowing.parse_ts_from_name(path.name)
        return ts is not None and ts.strftime("%Y%m%d") in self._dates

    def find_unmasked_imgages(self, cam_in_path: Path, masked_cam_out_path: Path) -> List[Path]:
        source_images = sorted(cam_in_path.glob("*.[pj][np][ge]*"))
        # Day-shard filter (load-bearing for concurrency): only mask THIS task's
        # day(s), so concurrent (cam, day) tasks writing the shared masked dir
        # never target the same source frame.
        if self._dates is not None:
            source_images = [p for p in source_images if self._in_dates(p)]
        masked_images = set(f.name.replace("masked_", "") for f in masked_cam_out_path.glob("masked_*"))
        unmasked_images = [img for img in source_images if img.name not in masked_images]
        return unmasked_images

    def create_output_dir(self) -> dict[str, dict[str, Path]]:
        output_dir_dict = {}
        tag = self.config_tag()
        for key in self.frame_dirs_per_cam.keys():
            # Masking is independent of the interval/window config, so the masked
            # frames are shared across configs (one dir per camera).
            masked_img_dir: Path = self.output_path / "masked" / key
            Path.mkdir(masked_img_dir, parents=True, exist_ok=True)
            # Backgrounds are config-specific: encode the tag in the path so
            # different configs are distinct, comparable products.
            background_img_dir: Path = self.output_path / key / tag
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

    # ------------------------------------------------------------------ #
    # Time-based subsampling / windowing (shared logic in windowing.py)
    # ------------------------------------------------------------------ #
    def _subsample_by_interval(self, paths: list[Path]) -> list[Path]:
        """Greedily keep one frame per ``frame_interval_sec`` by timestamp."""
        interval = self._config.frame_interval_sec
        if not interval:
            return paths
        kept_names = set(windowing.select_by_interval([p.name for p in paths], interval))
        return [p for p in paths if p.name in kept_names]

    def process_windowed_backgrounds(
        self,
        masked_img_dir: Path,
        background_img_dir: Path,
        tile_size=(512, 512),
        min_frames: int = 3,
    ) -> None:
        """Produce one background per time window from the masked frames.

        Frames are optionally subsampled by ``frame_interval_sec`` and grouped
        into ``background_window`` buckets; each bucket yields one background
        named ``background_<window-start>.png``. Existing outputs are skipped so
        the stage is resumable and a repeat config does no work.
        """
        masked_images = self._find_images_by_path(masked_img_dir, role="masked")
        if not masked_images:
            print(f"No masked images found in {masked_img_dir}")
            return

        # Restrict to this task's day(s) even though the masked dir is shared
        # across days, so only this day's window(s) are produced.
        if self._dates is not None:
            masked_images = [p for p in masked_images if self._in_dates(p)]

        masked_images = self._subsample_by_interval(masked_images)

        # Group frames by window start (preserve chronological order).
        win_cfg = self._config.background_window
        windows: "OrderedDict[datetime, list[Path]]" = OrderedDict()
        for p in masked_images:
            ts = windowing.parse_ts_from_name(p.name)
            if ts is None:
                continue
            windows.setdefault(windowing.window_bucket(ts, win_cfg), []).append(p)

        print(f"{len(windows)} background window(s) for config '{self.config_tag()}'")
        for bucket_start, frames in windows.items():
            out_name = f"background_{windowing.ts_to_name(bucket_start)}.png"
            out_path = background_img_dir / out_name
            if out_path.exists():
                continue  # resume / skip already-produced window
            self._compute_window_background(frames, out_path, tile_size=tile_size, min_frames=min_frames)

    def _compute_window_background(
        self,
        masked_paths: list[Path],
        out_path: Path,
        tile_size=(512, 512),
        min_frames: int = 3,
    ) -> bool:
        """Tile-wise median over all (bee-masked) frames in one window."""
        if len(masked_paths) < min_frames:
            print(f"Window {out_path.name}: only {len(masked_paths)} frames (< {min_frames}); skipping")
            return False

        first_img = self._read_image(masked_paths[0])
        if first_img is None:
            print(f"Could not read {masked_paths[0]}")
            return False
        H, W = first_img.shape
        n = len(masked_paths)

        memmap_file = self._new_memmap_path("win")
        stack = np.memmap(memmap_file, dtype="uint8", mode="w+", shape=(n, H, W))
        for k, p in enumerate(masked_paths):
            img = self._read_image(p)
            stack[k, :, :] = img if (img is not None and img.shape == (H, W)) else 0
        stack.flush()
        del stack
        gc.collect()

        stack = np.memmap(memmap_file, dtype="uint8", mode="r", shape=(n, H, W))
        background = np.zeros((H, W), dtype=np.uint8)
        results = Parallel(n_jobs=8)(
            delayed(self._process_tile_stack)(stack, i, j, tile_size, True)
            for i in range(0, H, tile_size[0])
            for j in range(0, W, tile_size[1])
        )
        for i, i_end, j, j_end, tile_result in results:
            background[i:i_end, j:j_end] = tile_result

        background = self._apply_clahe(background)
        self._save_image(background, out_path)
        print("Window background saved to:", out_path)

        del stack
        gc.collect()
        try:
            memmap_file.unlink()
        except Exception as e:
            print(f"Could not delete memmap file: {e}")
        return True

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

        # Optional time-based subsampling of the input frames.
        masked_images = self._subsample_by_interval(masked_images)

        background_images = self._find_images_by_path(background_img_dir, role="background")

        image_queue: Deque[tuple[np.ndarray, Path]] = deque()

        last_processed_img_name = (
            background_images[-1].name.replace("background", "masked") if background_images else None
        )

        start_idx = 0
        if last_processed_img_name:
            try:
                masked_names = [p.name for p in masked_images]
                last_index = masked_names.index(last_processed_img_name)
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

        self._memmap_file = self._new_memmap_path("rolling")
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
        # Atomic write: encode in memory, write to a per-pid temp file in the same
        # directory, then os.replace() onto the final path. A concurrent/duplicate
        # (cam, day) task can therefore never observe or produce a torn PNG.
        output_path = Path(output_path)
        ext = output_path.suffix or ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for {output_path}")
        tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(str(tmp_path), str(output_path))
