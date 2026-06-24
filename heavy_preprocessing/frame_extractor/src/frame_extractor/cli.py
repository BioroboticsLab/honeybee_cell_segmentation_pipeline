import argparse
from pathlib import Path

from frame_extractor.global_video_processor import GlobalVideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files at regular intervals.")
    parser.add_argument("base_dir", type=str, help="Path to the base directory containing video folders.")
    parser.add_argument("out_dir", type=str, help="Path to the output directory for extracted frames.")
    parser.add_argument("--file_format", type=str, default="png", help="Image file format for output (default: png).")
    parser.add_argument("--interval_in_sec", type=int, default=5, help="Interval in seconds between frames (default: 5).")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of parallel workers (default: 2).")
    parser.add_argument("--fps", type=int, default=3, help="FPS rate of the videos (default: 3).")
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Optional YYYYMMDD day folders to restrict to (default: all).",
    )
    parser.add_argument(
        "--cams", nargs="+", default=None,
        help="Optional cam-N folders to restrict to (default: all).",
    )
    parser.add_argument(
        "--decoder", type=str, default="hevc_cuvid",
        help="ffmpeg -c:v decoder (default: hevc_cuvid for NVIDIA NVDEC). "
             "Pass 'none' for software decode on CPU-only nodes.",
    )
    parser.add_argument(
        "--ffmpeg-bin", type=str, default=None,
        help="Path to the ffmpeg binary (default: bundled bin/ then PATH).",
    )

    args = parser.parse_args()

    decoder = None if (args.decoder or "").lower() == "none" else args.decoder

    processor = GlobalVideoProcessor(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        file_format=args.file_format,
        interval_in_sec=args.interval_in_sec,
        max_workers=args.max_workers,
        fps=args.fps,
        dates=args.dates,
        cams=args.cams,
        decoder=decoder,
        ffmpeg_bin_path=args.ffmpeg_bin,
    )
    processor.run()


if __name__ == "__main__":
    main()
