"""Tests for the deterministic frame-naming logic.

These cover the property bb_hpc relies on for skip checks: a coarser interval
that is a multiple of a finer one produces a subset of the finer run's output
filenames, so a per-filename existence check finds them already done.
"""
from pathlib import Path

from frame_extractor.naming import (
    expected_frame_filenames,
    expected_frame_names_for_video,
    per_video_step,
    selected_video_indices,
)


def _write_minute_video_txt(dir_path: Path, idx: int, fps: int = 3, seconds: int = 60) -> Path:
    """Create a fake .txt sidecar with one timestamp line per frame (fps*seconds)."""
    txt = dir_path / f"cam-0_video{idx:03d}.txt"
    lines = []
    for s in range(seconds):
        for f in range(fps):
            lines.append(f"2025060{idx % 10}T{s:02d}{f:02d}00.000000.000Z")
    txt.write_text("\n".join(lines) + "\n")
    return txt


def test_per_video_step_and_indices():
    # interval <= 60 s: every video processed, stride = interval*fps
    assert per_video_step(30, 3) == 90
    assert list(selected_video_indices(10, 30)) == list(range(10))
    # interval > 60 s: one frame per processed video, every N-th video (N = interval//60)
    assert per_video_step(600, 3) == 180
    assert list(selected_video_indices(20, 300)) == [0, 5, 10, 15]
    assert list(selected_video_indices(20, 600)) == [0, 10]


def test_coarser_multiple_is_subset(tmp_path):
    fps = 3
    txts = [_write_minute_video_txt(tmp_path, i, fps=fps) for i in range(20)]

    names_5min = expected_frame_filenames(txts, interval_in_sec=300, fps=fps)
    names_10min = expected_frame_filenames(txts, interval_in_sec=600, fps=fps)

    # 10-min frames must be a strict subset of 5-min frames (the key property).
    assert names_10min
    assert names_10min < names_5min


def test_one_frame_per_processed_minute_video(tmp_path):
    # A 1-minute video at interval >= 60 yields exactly its first timestamp.
    txt = _write_minute_video_txt(tmp_path, 0, fps=3)
    names = expected_frame_names_for_video(txt, interval_in_sec=60, fps=3)
    assert len(names) == 1
    assert names[0].endswith(".png")
