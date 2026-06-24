import os
import subprocess
import platform
from pathlib import Path
import shutil
import tempfile
from logging import Logger

from frame_extractor.naming import read_timestamps


class FrameExtractor:
    def __init__(
        self,
        logger: Logger,
        interval_sec: int = 60,
        fps: int = 3,
        file_format: str = "png",
        decoder: str | None = "hevc_cuvid",
        ffmpeg_bin_path: str | Path | None = None,
    ):
        self.logger = logger
        self.interval_sec = interval_sec
        self.video_fps = fps
        self.file_format = file_format
        # Decoder for ffmpeg's -c:v. Default is NVIDIA NVDEC (fast, requires a
        # GPU build of ffmpeg). Set to None for software decode (ffmpeg auto-
        # selects) so the extractor also runs on CPU-only nodes / plain ffmpeg.
        self.decoder = decoder
        self.ffmpeg_bin_path = self.determine_ffmpeg_bin_path(ffmpeg_bin_path)

    def determine_ffmpeg_bin_path(self, override: str | Path | None = None) -> Path:
        """Resolve the ffmpeg binary.

        Order: explicit override -> ``BB_FFMPEG_BIN`` env var -> binary bundled
        under this package's ``bin/`` -> ``ffmpeg`` on PATH (e.g. the conda env).
        Falling back to PATH lets the extractor run in environments (HPC conda,
        containers) that provide ffmpeg without a vendored binary.
        """
        if override:
            return Path(override)
        env_override = os.environ.get("BB_FFMPEG_BIN")
        if env_override:
            return Path(env_override)

        exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        bundled = Path(__file__).parent.resolve() / "bin" / exe
        if bundled.exists():
            return bundled

        on_path = shutil.which("ffmpeg")
        if on_path:
            return Path(on_path)

        # Return the bundled path anyway; using it later will raise a clear error.
        self.logger.warning(f"ffmpeg not found in bin/ or on PATH; defaulting to {bundled}")
        return bundled

    def read_timestamps(self, txt_file: Path) -> list[str]:
        # Kept for backwards compatibility; delegates to the shared helper so the
        # engine and external schedulers select identical timestamps.
        return read_timestamps(txt_file)

    def extract_from(self, video_file_path: Path, output_dir: Path) -> None:
        txt_file = video_file_path.with_suffix(".txt")

        if not video_file_path.exists():
            self.logger.error(f"Video file not found: {video_file_path}")
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        if not txt_file.exists():
            self.logger.error(f"Timestamp file not found: {txt_file}")
            raise FileNotFoundError(f"Timestamp file not found: {txt_file}")

        all_timestamps = self.read_timestamps(txt_file)
        step = self.interval_sec * self.video_fps
        selected_timestamps = all_timestamps[::step]
        frame_count = len(selected_timestamps)

        # Per-filename skip: if every expected output already exists, do no ffmpeg
        # work. This is what makes re-running a date at a coarser (multiple)
        # interval ~free -- those frames were already written by the finer run.
        targets = [output_dir / f"{ts}.{self.file_format}" for ts in selected_timestamps]
        if targets and all(p.exists() for p in targets):
            self.logger.info(
                f"Skipping {video_file_path.name}: all {frame_count} frames already extracted"
            )
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"tmp_{video_file_path.stem}_"))

        cmd = [str(self.ffmpeg_bin_path), "-y"]
        if self.decoder:
            cmd += ["-c:v", self.decoder]
        cmd += [
            "-i",
            str(video_file_path),
            "-vf",
            f"select='not(mod(n\\,{step}))'",
            "-vsync",
            "vfr",
            str(tmp_dir / f"frame_%05d.{self.file_format}"),
        ]

        # TODO: implement proper logging to file for errors
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        extracted_frames = sorted(tmp_dir.glob(f"frame_*.{self.file_format}"))
        if len(extracted_frames) != frame_count:
            self.logger.error(f"Mismatch: {len(extracted_frames)} frames vs {frame_count} timestamps")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Mismatch: {len(extracted_frames)} frames vs {frame_count} timestamps")

        for img_path, target in zip(extracted_frames, targets):
            shutil.move(str(img_path), str(target))

        shutil.rmtree(tmp_dir, ignore_errors=True)

        self.logger.info(f"Extracted {len(selected_timestamps)} frames from {video_file_path.name} to {output_dir}")
