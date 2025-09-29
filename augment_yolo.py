"""
Augment YOLO train set with flips and 90-degree rotations.

Reads from data/yolo/train/{images,labels} and writes augmented samples to
 data/yolo/train_aug/{images,labels}. Label transforms preserve YOLO format.

Augmentations:
- Horizontal flip
- Vertical flip
- Rotate 90, 180, 270 degrees

Note: Keep original images. You can later merge train_aug into train to increase dataset size.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo"
TRAIN_IMAGES = YOLO_ROOT / "train" / "images"
TRAIN_LABELS = YOLO_ROOT / "train" / "labels"
OUT_IMAGES = YOLO_ROOT / "train_aug" / "images"
OUT_LABELS = YOLO_ROOT / "train_aug" / "labels"


def ensure_dirs() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    items: List[Tuple[int, float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            items.append((cls, x, y, w, h))
    return items


def write_yolo_labels(label_path: Path, items: List[Tuple[int, float, float, float, float]]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in items:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def flip_h(items: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    out = []
    for cls, x, y, w, h in items:
        out.append((cls, clamp01(1.0 - x), y, w, h))
    return out


def flip_v(items: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    out = []
    for cls, x, y, w, h in items:
        out.append((cls, x, clamp01(1.0 - y), w, h))
    return out


def rot90(items: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    # (x, y) -> (y, 1 - x)
    out = []
    for cls, x, y, w, h in items:
        out.append((cls, clamp01(y), clamp01(1.0 - x), h, w))
    return out


def rot180(items: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    # (x, y) -> (1 - x, 1 - y)
    out = []
    for cls, x, y, w, h in items:
        out.append((cls, clamp01(1.0 - x), clamp01(1.0 - y), w, h))
    return out


def rot270(items: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    # (x, y) -> (1 - y, x)
    out = []
    for cls, x, y, w, h in items:
        out.append((cls, clamp01(1.0 - y), clamp01(x), h, w))
    return out


def apply_and_save(img_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> None:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # Horizontal flip
    img_h = cv2.flip(img, 1)
    cv2.imwrite(str(OUT_IMAGES / f"{stem}_fh.jpg"), img_h)
    write_yolo_labels(OUT_LABELS / f"{stem}_fh.txt", flip_h(labels))

    # Vertical flip
    img_v = cv2.flip(img, 0)
    cv2.imwrite(str(OUT_IMAGES / f"{stem}_fv.jpg"), img_v)
    write_yolo_labels(OUT_LABELS / f"{stem}_fv.txt", flip_v(labels))

    # Rotate 90
    img_r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(OUT_IMAGES / f"{stem}_r90.jpg"), img_r90)
    write_yolo_labels(OUT_LABELS / f"{stem}_r90.txt", rot90(labels))

    # Rotate 180
    img_r180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(str(OUT_IMAGES / f"{stem}_r180.jpg"), img_r180)
    write_yolo_labels(OUT_LABELS / f"{stem}_r180.txt", rot180(labels))

    # Rotate 270
    img_r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(OUT_IMAGES / f"{stem}_r270.jpg"), img_r270)
    write_yolo_labels(OUT_LABELS / f"{stem}_r270.txt", rot270(labels))


def main() -> int:
    ensure_dirs()
    image_paths = sorted((p for p in TRAIN_IMAGES.glob("*.jpg")))
    for img_path in image_paths:
        labels = read_yolo_labels(TRAIN_LABELS / f"{img_path.stem}.txt")
        if not labels:
            # skip empty labels to avoid propagating noise
            continue
        apply_and_save(img_path, labels)
    print(f"Augmented images written to: {OUT_IMAGES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
