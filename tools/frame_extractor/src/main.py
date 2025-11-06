import argparse
from pathlib import Path
from global_video_processor import GlobalVideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files at regular intervals.")
    parser.add_argument("base_dir", type=str, help="Path to the base directory containing video folders.")
    parser.add_argument("out_dir", type=str, help="Path to the output directory for extracted frames.")
    parser.add_argument("--file_format", type=str, default="png", help="Image file format for output (default: png).")
    parser.add_argument("--interval_in_sec", type=int, default=5, help="Interval in seconds between frames (default: 5).")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of parallel workers (default: 2).")
    parser.add_argument("--fps", type=int, default=3, help="FPS rate of the videos (default: 3).")

    args = parser.parse_args()

    processor = GlobalVideoProcessor(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        file_format=args.file_format,
        interval_in_sec=args.interval_in_sec,
        max_workers=args.max_workers,
        fps=args.fps,
    )
    processor.run()


if __name__ == "__main__":
    main()
