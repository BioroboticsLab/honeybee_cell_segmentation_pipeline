"""Pure, dependency-light helpers for background frame selection and naming.

Kept free of heavy imports (no torch / cupy / cv2) so an external scheduler
(e.g. bb_hpc) can import these to compute the exact set of background output
filenames a given configuration will produce -- the basis for per-filename skip
checks -- without pulling in the GPU stack.

The background-generation engine imports the same functions so the engine and
the scheduler always agree on timestamps, windows, output names, and the
config tag used in the output path.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

# Matches the frame timestamp embedded in extracted / masked / background
# filenames, e.g. masked_20250603T120000.000000.000Z.png -> 20250603T120000
_TS_RE = re.compile(r"(\d{8}T\d{6})")


def parse_ts_from_name(name: str) -> datetime | None:
    m = _TS_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def ts_to_name(ts: datetime) -> str:
    """Format a window-start timestamp like the frame timestamps."""
    return ts.strftime("%Y%m%dT%H%M%S") + ".000000.000Z"


def window_bucket(ts: datetime, background_window: str | int) -> datetime:
    """Floor a timestamp to the start of its background window."""
    if background_window == "hour":
        return ts.replace(minute=0, second=0, microsecond=0)
    if background_window == "day":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = int(background_window)
    day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    offset = int((ts - day_start).total_seconds())
    return day_start + timedelta(seconds=(offset // seconds) * seconds)


def config_tag(
    frame_interval_sec: int | None,
    background_window: str | int | None,
    window_size: int,
    num_median_images: int,
) -> str:
    """Short tag identifying a background configuration (used in the output path)."""
    interval = frame_interval_sec or 0
    if background_window:
        return f"int{interval}s_win{background_window}"
    return f"count_w{window_size}_n{num_median_images}_int{interval}s"


def select_by_interval(names_sorted: list[str], frame_interval_sec: int | None) -> list[str]:
    """Greedily keep one frame name per ``frame_interval_sec`` by timestamp.

    ``names_sorted`` must be ordered chronologically (the frame timestamp format
    is lexically sortable, so sorting by filename works).
    """
    if not frame_interval_sec:
        return list(names_sorted)
    kept: list[str] = []
    last_ts: datetime | None = None
    for name in names_sorted:
        ts = parse_ts_from_name(name)
        if ts is None:
            continue
        if last_ts is None or (ts - last_ts).total_seconds() >= frame_interval_sec:
            kept.append(name)
            last_ts = ts
    return kept


def expected_background_names(
    frame_names: list[str],
    frame_interval_sec: int | None,
    background_window: str | int,
) -> set[str]:
    """Background output filenames a windowed run would produce from these frames.

    ``frame_names`` are the extracted-frame (or masked) filenames for one camera.
    Mirrors the engine: subsample by interval, bucket by window, one background
    per bucket named ``background_<window-start>.png``.
    """
    names = sorted(frame_names)
    names = select_by_interval(names, frame_interval_sec)
    buckets: set[datetime] = set()
    for name in names:
        ts = parse_ts_from_name(name)
        if ts is None:
            continue
        buckets.add(window_bucket(ts, background_window))
    return {f"background_{ts_to_name(b)}.png" for b in buckets}
