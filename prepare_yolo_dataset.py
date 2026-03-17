"""
Prepare a merged YOLO dataset from multiple sources.

Sources (in priority order):
  1. data/*.jpg  +  data/labels/*.txt   (your own annotated chips — class 0 = "chip")
  2. data/roboflow/<project>/           (Roboflow downloads, optional via --extra-source)

Actions:
  - Collect all image/label pairs from every source.
  - Remap class IDs so they are consistent across sources.
  - Split into train/val (80/20) deterministically.
  - Write dataset.yaml with the merged class list.

Usage:
    # Own images only (single-class "chip"):
    python prepare_yolo_dataset.py

    # Own images + Roboflow data (multi-class):
    python prepare_yolo_dataset.py --extra-source data/roboflow

    # Force specific class names (comma-separated, in class-id order):
    python prepare_yolo_dataset.py --names "chip,orange,pink,green,black,white"
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_SRC_DIR = DATA_DIR / "labels"

YOLO_ROOT = DATA_DIR / "yolo"
TRAIN_IMAGES = YOLO_ROOT / "train" / "images"
TRAIN_LABELS = YOLO_ROOT / "train" / "labels"
VAL_IMAGES = YOLO_ROOT / "val" / "images"
VAL_LABELS = YOLO_ROOT / "val" / "labels"


# ──────────────────────────────────────────────
# YAML helpers
# ──────────────────────────────────────────────

def _parse_yaml_names(yaml_path: Path) -> List[str]:
    """Extract class names from a Roboflow-style data.yaml."""
    text = yaml_path.read_text(encoding="utf-8")
    # Handle both  names: [a, b, c]  and  names:\n  - a\n  - b
    list_match = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if list_match:
        return [n.strip().strip("'\"") for n in list_match.group(1).split(",")]
    block_match = re.findall(r"^[ \t]*-[ \t]+(.+?)$", text, re.MULTILINE)
    if block_match:
        return [n.strip().strip("'\"") for n in block_match]
    return []


# ──────────────────────────────────────────────
# Image / label collection
# ──────────────────────────────────────────────

def _collect_own(
    global_names: List[str],
) -> List[Tuple[Path, Path, Dict[int, int]]]:
    """
    Collect (image_path, label_path, class_id_remap) tuples from data/*.jpg.
    Own annotations use class 0 only; mapped to global_names.index("chip").
    """
    own_class_id = global_names.index("chip") if "chip" in global_names else 0

    images = sorted(
        p for p in DATA_DIR.glob("*.jpg")
        if "debug" not in p.stem.lower()
    )
    pairs = []
    for img in images:
        label = LABELS_SRC_DIR / f"{img.stem}.txt"
        remap = {0: own_class_id}
        pairs.append((img, label, remap))
    return pairs


def _collect_roboflow(
    source_dir: Path,
    global_names: List[str],
) -> List[Tuple[Path, Path, Dict[int, int]]]:
    """
    Walk source_dir for Roboflow project folders and collect pairs.
    Reads each project's data.yaml to build a local→global class remap.
    """
    pairs = []
    for project_dir in sorted(source_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        yaml_path = project_dir / "data.yaml"
        if not yaml_path.exists():
            continue
        local_names = _parse_yaml_names(yaml_path)
        if not local_names:
            print(f"  WARNING: Could not parse class names from {yaml_path}")
            continue

        # Build local class_id → global class_id
        remap: Dict[int, int] = {}
        for local_id, name in enumerate(local_names):
            if name in global_names:
                remap[local_id] = global_names.index(name)
            else:
                # Unknown class — append to global list and remap
                global_names.append(name)
                remap[local_id] = len(global_names) - 1

        # Collect train + valid images
        for split in ("train", "valid", "val"):
            img_dir = project_dir / split / "images"
            lbl_dir = project_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                label = lbl_dir / f"{img.stem}.txt"
                pairs.append((img, label, remap))

        print(f"  Roboflow project '{project_dir.name}': "
              f"{len(local_names)} classes, added pairs from {project_dir}")
    return pairs


# ──────────────────────────────────────────────
# Label read / write
# ──────────────────────────────────────────────

def _read_label(label_path: Path, remap: Dict[int, int]) -> List[str]:
    """Read YOLO label lines and apply class ID remap. Strips confidence column."""
    if not label_path.exists():
        return []
    lines_out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        mapped_cls = remap.get(cls, cls)
        lines_out.append(f"{mapped_cls} " + " ".join(parts[1:5]))
    return lines_out


# ──────────────────────────────────────────────
# Split + copy
# ──────────────────────────────────────────────

def _deterministic_split(
    pairs: List[Tuple[Path, Path, Dict[int, int]]],
    val_ratio: float = 0.2,
) -> Tuple[List, List]:
    total = len(pairs)
    val_count = max(1, int(round(total * val_ratio))) if total > 0 else 0
    train = pairs[:-val_count] if val_count else pairs
    val = pairs[-val_count:] if val_count else []
    return train, val


def _copy_pairs(
    pairs: List[Tuple[Path, Path, Dict[int, int]]],
    dst_images: Path,
    dst_labels: Path,
) -> int:
    copied = 0
    seen_stems: Dict[str, int] = {}

    for img_path, label_path, remap in pairs:
        # Deduplicate stem collisions across sources
        stem = img_path.stem
        if stem in seen_stems:
            seen_stems[stem] += 1
            stem = f"{stem}_{seen_stems[stem]}"
        else:
            seen_stems[stem] = 0

        suffix = img_path.suffix
        shutil.copy2(img_path, dst_images / f"{stem}{suffix}")

        label_lines = _read_label(label_path, remap)
        (dst_labels / f"{stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        copied += 1
    return copied


# ──────────────────────────────────────────────
# dataset.yaml
# ──────────────────────────────────────────────

def _write_dataset_yaml(names: List[str]) -> Path:
    yaml_path = YOLO_ROOT / "dataset.yaml"
    names_str = ", ".join(f"'{n}'" for n in names)
    content = (
        f"path: {YOLO_ROOT.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(names)}\n"
        f"names: [{names_str}]\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare merged YOLO dataset.")
    parser.add_argument(
        "--extra-source", type=str, default=None,
        help="Path to a directory of Roboflow project folders (e.g. data/roboflow)",
    )
    parser.add_argument(
        "--names", type=str, default=None,
        help="Comma-separated class names in class-id order "
             "(default: auto-discovered from sources)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of data reserved for validation (default: 0.2)",
    )
    args = parser.parse_args()

    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    # Seed global class list
    if args.names:
        global_names: List[str] = [n.strip() for n in args.names.split(",")]
    else:
        global_names = ["chip"]  # default single-class for own annotations

    # Collect own images
    own_pairs = _collect_own(global_names)
    print(f"Own annotated images: {len(own_pairs)}")

    # Collect Roboflow images (mutates global_names to add discovered classes)
    extra_pairs: List[Tuple[Path, Path, Dict[int, int]]] = []
    if args.extra_source:
        extra_dir = Path(args.extra_source)
        if extra_dir.exists():
            extra_pairs = _collect_roboflow(extra_dir, global_names)
            print(f"Roboflow images: {len(extra_pairs)}")
        else:
            print(f"WARNING: --extra-source path not found: {extra_dir}")

    all_pairs = own_pairs + extra_pairs
    if not all_pairs:
        print("No images found. Nothing to do.")
        return 0

    train_pairs, val_pairs = _deterministic_split(all_pairs, args.val_ratio)

    n_train = _copy_pairs(train_pairs, TRAIN_IMAGES, TRAIN_LABELS)
    n_val   = _copy_pairs(val_pairs, VAL_IMAGES, VAL_LABELS)
    yaml_path = _write_dataset_yaml(global_names)

    print(f"\nPrepared YOLO dataset at: {YOLO_ROOT}")
    print(f"  Train: {n_train}  Val: {n_val}")
    print(f"  Classes ({len(global_names)}): {global_names}")
    print(f"  Dataset yaml: {yaml_path}")
    print("\nTrain command:")
    print(f"  yolo detect train model=yolov8n.pt data={yaml_path} "
          "imgsz=640 epochs=100 batch=16 workers=0 amp=False")
    print("\nExport CoreML:")
    print("  yolo export model=runs/detect/train/weights/best.pt "
          "format=coreml imgsz=640")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
