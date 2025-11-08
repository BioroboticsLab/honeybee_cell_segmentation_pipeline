import argparse
import sys
from pathlib import Path
from background_generator.background_img_generator import BackgroundImageGenerator
from background_generator.utils import BgImageGenConfig
from global_video_processor import GlobalVideoProcessor


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate bee-free background images from video frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main arguments for background generator
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to directory containing extracted frames (organized by camera)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for background images"
    )

    # Frame extraction subcommand
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from video before generating backgrounds"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing video files (required if --extract-frames is set)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Frame extraction interval in seconds (must be divisor of 60 if <60, or multiple of 60 if >=60)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for frame extraction"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Frames per second for video processing"
    )

    # Background generation configuration
    bg_group = parser.add_argument_group("background generation settings")
    bg_group.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of consecutive frames for rolling median computation"
    )
    bg_group.add_argument(
        "--num-median-images",
        type=int,
        default=200,
        help="Number of rolling median images to compute per background"
    )
    bg_group.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of background images to create per camera (None = unlimited)"
    )
    bg_group.add_argument(
        "--jump-size",
        type=int,
        default=1,
        help="Step size to skip from last processed image"
    )
    bg_group.add_argument(
        "--apply-clahe",
        type=str,
        choices=["intermediate", "post"],
        default="post",
        help="When to apply CLAHE contrast enhancement"
    )
    bg_group.add_argument(
        "--mask-dilation",
        type=int,
        choices=[0, 9, 15, 25],
        default=15,
        help="Morphological dilation kernel size for masks (0 = disabled)"
    )
    bg_group.add_argument(
        "--median-computation",
        type=str,
        choices=["cupy", "cuda_support", "masked_array"],
        default="cupy",
        help="Method for median computation (cupy = fastest GPU, masked_array = CPU)"
    )
    bg_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Processing device for segmentation and computation"
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.extract_frames and not args.video_dir:
        raise ValueError("--video-dir is required when --extract-frames is set")

    if args.extract_frames:
        if not args.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
        if not args.video_dir.is_dir():
            raise NotADirectoryError(f"Video path is not a directory: {args.video_dir}")

    if not args.source.exists():
        if args.extract_frames:
            print(f"Source directory will be created: {args.source}")
            args.source.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Source directory not found: {args.source}")

    if not args.output.exists():
        print(f"Output directory will be created: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)


def extract_frames(args: argparse.Namespace) -> None:
    processor = GlobalVideoProcessor(
        base_dir=args.video_dir,
        out_dir=args.source,
        interval_in_sec=args.interval,
        max_workers=args.max_workers,
        fps=args.fps
    )
    processor.run()


def generate_backgrounds(args: argparse.Namespace) -> None:

    config = BgImageGenConfig(
        window_size=args.window_size,
        num_median_images=args.num_median_images,
        max_cycles=args.max_cycles,
        jump_size_from_last=args.jump_size,
        apply_clahe=args.apply_clahe,
        mask_dilation=args.mask_dilation,
        median_computation=args.median_computation,
        device=args.device
    )

    generator = BackgroundImageGenerator(
        source_path=args.source,
        output_path=args.output,
        config=config
    )
    generator.run()


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        if args.extract_frames:
            extract_frames(args)

        generate_backgrounds(args)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
