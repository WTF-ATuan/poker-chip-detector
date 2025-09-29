"""
Prepare YOLO dataset from data/*.jpg and data/labels/*.txt.

Actions:
- Split into train/val (80/20) deterministically by filename.
- Copy images and create paired label files.
- Strip confidence column from labels (keep: class x y w h).
- Write dataset.yaml to data/yolo/dataset.yaml.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_SRC_DIR = DATA_DIR / "labels"

YOLO_ROOT = DATA_DIR / "yolo"
TRAIN_IMAGES = YOLO_ROOT / "train" / "images"
TRAIN_LABELS = YOLO_ROOT / "train" / "labels"
VAL_IMAGES = YOLO_ROOT / "val" / "images"
VAL_LABELS = YOLO_ROOT / "val" / "labels"


def list_images() -> List[Path]:
    images = [Path(p) for p in glob.glob(str(DATA_DIR / "*.jpg"))]
    # exclude debug renders
    images = [p for p in images if "debug" not in p.name.lower()]
    return sorted(images)


def read_and_strip_confidence(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    lines_out: List[str] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # Keep first 5 columns: class x y w h (normalized)
            parts5 = parts[:5]
            # Validate lengths
            if len(parts5) < 5:
                continue
            lines_out.append(" ".join(parts5))
    return lines_out


def ensure_dirs() -> None:
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)


def deterministic_split(paths: List[Path], val_ratio: float = 0.2) -> Tuple[List[Path], List[Path]]:
    # Deterministic by filename sort: last N go to val
    total = len(paths)
    val_count = max(1, int(round(total * val_ratio))) if total > 0 else 0
    train = paths[:-val_count] if val_count > 0 else paths
    val = paths[-val_count:] if val_count > 0 else []
    return train, val


def copy_and_write_labels(images: List[Path], dst_images_dir: Path, dst_labels_dir: Path) -> None:
    for img_path in images:
        stem = img_path.stem
        dst_img = dst_images_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        src_label = LABELS_SRC_DIR / f"{stem}.txt"
        dst_label = dst_labels_dir / f"{stem}.txt"
        stripped_lines = read_and_strip_confidence(src_label)
        with open(dst_label, "w", encoding="utf-8") as f:
            for line in stripped_lines:
                f.write(line + "\n")


def write_dataset_yaml() -> Path:
    yaml_path = YOLO_ROOT / "dataset.yaml"
    yaml_content = (
        "path: data/yolo\n"
        "train: train/images\n"
        "val: val/images\n"
        "names: [chip]\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def main() -> int:
    ensure_dirs()
    images = list_images()
    if not images:
        print("No images found in data/*.jpg")
        return 0
    train_imgs, val_imgs = deterministic_split(images, val_ratio=0.2)

    copy_and_write_labels(train_imgs, TRAIN_IMAGES, TRAIN_LABELS)
    copy_and_write_labels(val_imgs, VAL_IMAGES, VAL_LABELS)
    yaml_path = write_dataset_yaml()

    print(f"Prepared YOLO dataset at: {YOLO_ROOT}")
    print(f" Train images: {len(train_imgs)}  Val images: {len(val_imgs)}")
    print(f" Dataset yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


