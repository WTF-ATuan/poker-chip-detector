"""
Download poker chip datasets from Roboflow Universe and merge them into
data/roboflow/ ready for use with prepare_yolo_dataset.py.

Datasets downloaded:
  1. Game Chips by Dorna        — 201 images, 5 classes, CC BY 4.0
     workspace: dorna / project: game-chips
  2. Poker-Chips Detector        — 95 images, 3 classes, CC BY 4.0
     workspace: urop2023-ar-metaverse / project: poker-chips-detector

Prerequisites:
    pip install roboflow          (already in requirements.txt)
    Sign up at https://roboflow.com (free) and copy your API key.

Usage:
    source .venv311/bin/activate
    python download_roboflow_dataset.py --api-key <YOUR_KEY>

    # After download, merge into data/ and run prepare_yolo_dataset.py:
    python prepare_yolo_dataset.py --extra-source data/roboflow

Output layout:
    data/roboflow/
        game-chips/          ← YOLOv8 format
            train/images/
            train/labels/
            valid/images/
            valid/labels/
            data.yaml
        poker-chips-detector/
            train/images/
            ...
            data.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _check_roboflow() -> None:
    try:
        import roboflow  # noqa: F401
    except ImportError:
        print("roboflow package not found. Install it with:\n  pip install roboflow")
        sys.exit(1)


DATASETS = [
    {
        "workspace": "dorna",
        "project": "game-chips",
        "version": 1,
        "description": "Game Chips by Dorna — 201 images, 5 classes",
    },
    {
        "workspace": "urop2023-ar-metaverse",
        "project": "poker-chips-detector",
        "version": 1,
        "description": "Poker-Chips Detector — 95 images, 3 classes",
    },
]

OUT_DIR = Path("data/roboflow")


def download_dataset(api_key: str, entry: dict) -> Path:
    from roboflow import Roboflow  # type: ignore

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(entry["workspace"]).project(entry["project"])
    version = project.version(entry["version"])

    out_path = OUT_DIR / entry["project"]
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {entry['description']}")
    print(f"  → {out_path}/")
    version.download("yolov8", location=str(out_path))
    return out_path


def print_class_summary(out_path: Path) -> None:
    """Read data.yaml and print discovered class names."""
    yaml_path = out_path / "data.yaml"
    if not yaml_path.exists():
        return
    import re
    text = yaml_path.read_text()
    names_match = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if names_match:
        names = [n.strip().strip("'\"") for n in names_match.group(1).split(",")]
        print(f"  Classes ({len(names)}): {names}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download poker chip datasets from Roboflow Universe."
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Your Roboflow API key (free account at https://roboflow.com)"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["game-chips", "poker-chips-detector", "all"],
        default=["all"],
        help="Which datasets to download (default: all)"
    )
    args = parser.parse_args()

    _check_roboflow()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = DATASETS
    if "all" not in args.datasets:
        selected = [d for d in DATASETS if d["project"] in args.datasets]

    for entry in selected:
        try:
            out_path = download_dataset(args.api_key, entry)
            print_class_summary(out_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

    print(f"\nAll downloads saved to: {OUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. Review class names in each data/roboflow/<project>/data.yaml")
    print("  2. Run:  python prepare_yolo_dataset.py --extra-source data/roboflow")
    print("  3. Run:  yolo detect train model=yolov8n.pt data=data/yolo/dataset.yaml \\")
    print("               imgsz=640 epochs=100 batch=16 workers=0 amp=False")
    print("  4. Export CoreML:")
    print("       yolo export model=runs/detect/train/weights/best.pt format=coreml imgsz=640")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
