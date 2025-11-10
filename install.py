import subprocess
import sys
from pathlib import Path
import math

INSTALL_GRAPH = {
    # Packages (for standalone installation)
    "honeybee-segmentor": [
        "packages/honeybee_segmentor"
    ],
    "comb-limitor": [
        "packages/honeybee_segmentor",
        "packages/comb_limitor"
    ],
    # Tools
    "annotation-tool": [
        "tools/annotation_tool"
    ],
    "mask-writer": [
        "tools/mask_writer"
    ],
    "frame-extractor": [
        "tools/frame_extractor"
    ],
    "cell-finder": [
        "packages/honeybee_segmentor",
        "packages/comb_limitor",
        "tools/cell_finder"
    ],
    "background-generator": [
        "packages/honeybee_segmentor",
        "tools/frame_extractor",
        "tools/background_generator"
    ],
}

DESCRIPTIONS = {
    # Packages
    "honeybee-segmentor": "Core segmentation framework (package)",
    "comb-limitor": "Binary comb mask generation (package)",
    # Tools
    "annotation-tool": "Interactive annotation UI with napari",
    "mask-writer": "Convert annotations to segmentation masks",
    "frame-extractor": "Extract frames from video files at regular intervals",
    "cell-finder": "Finds as many cells in honeybee comb images as possible",
    "background-generator": "Generate bee-free background images from extracted frames",
}


def get_repo_root():
    return Path(__file__).parent.absolute()


def install_path(path):
    repo_root = get_repo_root()
    full_path = repo_root / path

    if not full_path.exists():
        print(f"Error: Path does not exist: {full_path}")
        return False

    print(f"\nInstalling {path}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(full_path)],
            check=True,
            cwd=repo_root
        )
        print(f"{path} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {path}: {e}")
        return False


def print_pytorch_cuda_info():
    info_strings = [
        " ",
        "GPU Acceleration (Optional)",
        " ", "=", " ",
        "If you want to use GPU acceleration for deep learning, install PyTorch with CUDA support:",
        " ",
        "pip uninstall torch torchvision -y",
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        " ",
        "For other CUDA versions, visit: https://pytorch.org/get-started/locally/",
        " ",
    ]
    len_max_string = len(max(info_strings, key=len))
    line_len = len_max_string + 10

    print("\n" + "=" * (line_len + 4))
    for str in info_strings:
        if str in ["=", " "]:
            final_string = str * line_len
        else:
            blanks_total = line_len - len(str)
            before = math.ceil(blanks_total / 2)
            after = math.floor(blanks_total / 2)
            final_string = " " * before + str + " " * after
        print("||" + final_string + "||")
    print("=" * (line_len + 4) + "\n")


def install_tool(tool_name):
    if tool_name not in INSTALL_GRAPH:
        print(f"Unknown tool: {tool_name}")
        list_tools()
        return False

    print(f"\nInstalling {tool_name}...")
    print(f"   {DESCRIPTIONS[tool_name]}")

    # Track what we've already installed to avoid duplicates
    installed = set()

    for path in INSTALL_GRAPH[tool_name]:
        if path not in installed:
            if not install_path(path):
                return False
            installed.add(path)

    print(f"\n {tool_name} installed successfully!")

    if "packages/honeybee_segmentor" in INSTALL_GRAPH[tool_name]:
        print_pytorch_cuda_info()

    return True


def list_tools():

    print("\n" + "=" * 70)
    print("Available Tools:")
    print("=" * 70)

    for name, desc in DESCRIPTIONS.items():
        deps = INSTALL_GRAPH[name]
        dep_info = ""
        if len(deps) > 1:
            internal_deps = [d.split('/')[-1] for d in deps[:-1]]
            dep_info = f" (requires: {', '.join(internal_deps)})"

        print(f"\n  {name}")
        print(f"    {desc}{dep_info}")

    print("\n" + "=" * 70)
    print("\nUsage:")
    print("  python install.py <tool-name>")
    print("  python install.py --all")
    print("\nExample:")
    print("  python install.py cell-finder")
    print("=" * 70 + "\n")


def main():
    if len(sys.argv) < 2:
        list_tools()
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list":
        list_tools()
        sys.exit(0)

    if arg == "--all":
        print("\n" + "=" * 70)
        print("Installing all tools...")
        print("=" * 70)

        success = True
        all_paths = set()

        for tool_name in INSTALL_GRAPH.keys():
            for path in INSTALL_GRAPH[tool_name]:
                all_paths.add(path)

        # Install each unique path once
        for path in sorted(all_paths):
            if not install_path(path):
                success = False

        if success:
            print("\n" + "All tools installed successfully!" + "\n")
            print_pytorch_cuda_info()
        else:
            print("\n" + "=" * 70)
            print("Some tools failed to install. Check the output above.")
            print("=" * 70 + "\n")
            sys.exit(1)

        sys.exit(0)

    tool = arg
    success = install_tool(tool)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
