"""Deterministic frame-output naming and timestamp selection.

Shared by the frame-extractor engine and any external scheduler (e.g. bb_hpc)
so both agree on exactly which output files a given (videos, interval, fps)
combination should produce.

This is what makes a coarser interval that is a multiple of a finer one
"already done" for free: a 10-min run selects a subset of the videos/timestamps
a 5-min run does, with identical output filenames, so a per-filename existence
check finds them all present and schedules no work.
"""
from __future__ import annotations

from pathlib import Path


def read_timestamps(txt_file: Path) -> list[str]:
    """Read non-empty, stripped timestamp lines from a video's .txt sidecar."""
    with Path(txt_file).open("r") as f:
        return [line.strip() for line in f if line.strip()]


def per_video_step(interval_in_sec: int, fps: int) -> int:
    """Frame stride within a single video.

    Mirrors GlobalVideoProcessor: intervals <= 60 s sample every video at the
    requested interval; intervals > 60 s sample one frame per *processed* video
    (the per-video interval is clamped to 60 s and whole videos are skipped
    instead -- see ``selected_video_indices``).
    """
    effective = interval_in_sec if interval_in_sec <= 60 else 60
    return effective * fps


def selected_video_indices(num_videos: int, interval_in_sec: int, start_idx: int = 0) -> range:
    """Indices of the videos (in sorted per-camera order) processed for this interval.

    For interval > 60 s only every N-th video (N = interval // 60) is processed,
    so a 10-min interval selects a subset of the videos a 5-min interval does.
    """
    if interval_in_sec <= 60:
        return range(start_idx, num_videos)
    n = interval_in_sec // 60
    return range(start_idx, num_videos, n)


def selected_timestamps(all_timestamps: list[str], interval_in_sec: int, fps: int) -> list[str]:
    """Timestamps kept from one video's full timestamp list at this interval/fps."""
    step = per_video_step(interval_in_sec, fps)
    return all_timestamps[::step]


def expected_frame_names_for_video(
    txt_file: Path, interval_in_sec: int, fps: int, file_format: str = "png"
) -> list[str]:
    """Output filenames a single video would produce at this interval/fps."""
    ts = selected_timestamps(read_timestamps(txt_file), interval_in_sec, fps)
    return [f"{t}.{file_format}" for t in ts]


def expected_frame_filenames(
    video_txt_files_sorted: list[Path],
    interval_in_sec: int,
    fps: int,
    file_format: str = "png",
) -> set[str]:
    """Full set of output frame filenames for a sorted per-camera video list.

    ``video_txt_files_sorted`` must be the ``.txt`` sidecars in the same sorted
    order the engine processes the videos (sorted by filename). Pass the full
    per-(date, camera) list to get the set of files a complete run would create.
    """
    names: set[str] = set()
    for idx in selected_video_indices(len(video_txt_files_sorted), interval_in_sec):
        names.update(
            expected_frame_names_for_video(
                video_txt_files_sorted[idx], interval_in_sec, fps, file_format
            )
        )
    return names
