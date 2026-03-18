"""
Build a color-first YOLO dataset for poker chips.

Goal:
- Convert mixed source labels (chip / denomination / unknown) into color classes.
- Keep object boxes, remap class IDs to target color names.
- Produce deterministic train/val split and a report.

Usage:
  source .venv311/bin/activate
  python build_color_dataset.py --extra-source data/roboflow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OWN_IMAGES_DIR = DATA_DIR
OWN_LABELS_DIR = DATA_DIR / "labels"

OUT_ROOT = DATA_DIR / "yolo_color"
TRAIN_IMAGES = OUT_ROOT / "train" / "images"
TRAIN_LABELS = OUT_ROOT / "train" / "labels"
VAL_IMAGES = OUT_ROOT / "val" / "images"
VAL_LABELS = OUT_ROOT / "val" / "labels"

DEFAULT_TARGET_NAMES = ["red", "pink", "green", "black"]
# Seed mapping for known source labels. You can override with --class-map.
DEFAULT_CLASS_MAP = {
    "chip": "",
    "1": "red",
    "2": "pink",
    "3": "green",
    "4": "black",
    "5": "red",
    "10": "pink",
    "100": "black",
    "500": "green",
}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path
    local_names: List[str]
    source_id: str


def parse_yaml_names(yaml_path: Path) -> List[str]:
    text = yaml_path.read_text(encoding="utf-8")
    list_match = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if list_match:
        return [n.strip().strip("'\"") for n in list_match.group(1).split(",")]
    block_match = re.findall(r"^[ \t]*-[ \t]+(.+?)$", text, re.MULTILINE)
    if block_match:
        return [n.strip().strip("'\"") for n in block_match]
    return []


def collect_own_pairs() -> List[Pair]:
    pairs: List[Pair] = []
    for img in sorted(OWN_IMAGES_DIR.glob("*.jpg")) + sorted(OWN_IMAGES_DIR.glob("*.jpeg")):
        if "debug" in img.stem.lower():
            continue
        label = OWN_LABELS_DIR / f"{img.stem}.txt"
        pairs.append(Pair(img, label, ["chip"], "own"))
    return pairs


def collect_roboflow_pairs(extra_source: Path) -> List[Pair]:
    pairs: List[Pair] = []
    for project_dir in sorted(extra_source.iterdir()):
        if not project_dir.is_dir():
            continue
        yaml_path = project_dir / "data.yaml"
        if not yaml_path.exists():
            continue
        local_names = parse_yaml_names(yaml_path)
        if not local_names:
            continue
        for split in ("train", "valid", "val"):
            img_dir = project_dir / split / "images"
            lbl_dir = project_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png")):
                pairs.append(Pair(img, lbl_dir / f"{img.stem}.txt", local_names, project_dir.name))
    return pairs


def parse_class_map(raw: str) -> Dict[str, str]:
    mapping = dict(DEFAULT_CLASS_MAP)
    if not raw.strip():
        return mapping
    for item in raw.split(","):
        if ":" not in item:
            continue
        src, dst = item.split(":", 1)
        mapping[src.strip().lower()] = dst.strip().lower()
    return mapping


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def color_from_bbox(image_bgr: np.ndarray, cx: float, cy: float, w: float, h: float) -> str:
    ih, iw = image_bgr.shape[:2]
    x1 = int(clamp01(cx - w / 2) * iw)
    y1 = int(clamp01(cy - h / 2) * ih)
    x2 = int(clamp01(cx + w / 2) * iw)
    y2 = int(clamp01(cy + h / 2) * ih)
    if x2 <= x1 or y2 <= y1:
        return "black"

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "black"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    sat_mask = s_ch > 35
    val_mask = v_ch > 35
    mask = sat_mask & val_mask
    if np.count_nonzero(mask) < 12:
        mask = np.ones_like(sat_mask, dtype=bool)

    h_med = float(np.median(h_ch[mask]))
    s_med = float(np.median(s_ch[mask]))
    v_med = float(np.median(v_ch[mask]))

    if s_med < 45 and v_med < 90:
        return "black"
    if h_med <= 12 or h_med >= 170:
        return "red"
    if 130 <= h_med <= 170:
        return "pink"
    if 35 <= h_med <= 95:
        return "green"

    # fallback by nearest prototype in HSV
    prototypes = {
        "red": np.array([0.0, 180.0, 170.0], dtype=np.float32),
        "pink": np.array([155.0, 90.0, 190.0], dtype=np.float32),
        "green": np.array([65.0, 130.0, 140.0], dtype=np.float32),
        "black": np.array([0.0, 20.0, 60.0], dtype=np.float32),
    }
    sample = np.array([h_med, s_med, v_med], dtype=np.float32)
    best = "black"
    best_dist = float("inf")
    for name, proto in prototypes.items():
        dh = min(abs(sample[0] - proto[0]), 180.0 - abs(sample[0] - proto[0])) * 2.0
        ds = abs(sample[1] - proto[1]) * 0.6
        dv = abs(sample[2] - proto[2]) * 0.3
        dist = float(dh + ds + dv)
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def split_bucket(key: str, val_ratio: float) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    ratio = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if ratio < val_ratio else "train"


def write_dataset_yaml(names: List[str]) -> Path:
    yaml_path = OUT_ROOT / "dataset.yaml"
    names_str = ", ".join(f"'{n}'" for n in names)
    yaml_path.write_text(
        (
            f"path: {OUT_ROOT.resolve()}\n"
            "train: train/images\n"
            "val: val/images\n"
            f"nc: {len(names)}\n"
            f"names: [{names_str}]\n"
        ),
        encoding="utf-8",
    )
    return yaml_path


def ensure_clean_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for d in (TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS):
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build color-first YOLO dataset.")
    parser.add_argument("--extra-source", type=str, default="data/roboflow", help="Roboflow dataset root.")
    parser.add_argument("--target-names", type=str, default="red,pink,green,black", help="Target color classes.")
    parser.add_argument(
        "--class-map",
        type=str,
        default="",
        help="Override/extend mapping, e.g. '10:pink,100:black,500:green,chip:'",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    args = parser.parse_args()

    target_names = [n.strip().lower() for n in args.target_names.split(",") if n.strip()]
    if not target_names:
        raise ValueError("No target names provided.")
    target_to_id = {name: idx for idx, name in enumerate(target_names)}
    class_map = parse_class_map(args.class_map)

    ensure_clean_dirs()

    pairs: List[Pair] = []
    pairs.extend(collect_own_pairs())

    extra_source = Path(args.extra_source)
    if extra_source.exists():
        pairs.extend(collect_roboflow_pairs(extra_source))
    else:
        print(f"WARNING: extra source not found: {extra_source}")

    if not pairs:
        print("No image/label pairs found.")
        return 0

    report = {
        "target_names": target_names,
        "class_map": class_map,
        "images_total": len(pairs),
        "objects_by_color": {n: 0 for n in target_names},
        "objects_inferred_by_pixels": 0,
        "objects_dropped": 0,
        "per_source_images": {},
    }

    seen_stems: Dict[str, int] = {}

    for pair in pairs:
        report["per_source_images"][pair.source_id] = report["per_source_images"].get(pair.source_id, 0) + 1
        image_bgr = cv2.imread(str(pair.image_path))
        if image_bgr is None:
            continue

        if not pair.label_path.exists():
            continue
        src_lines = pair.label_path.read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []
        for line in src_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                local_cls = int(parts[0])
                cx, cy, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                continue

            local_name = str(local_cls)
            if 0 <= local_cls < len(pair.local_names):
                local_name = pair.local_names[local_cls].strip().lower()

            mapped = class_map.get(local_name, "")
            if mapped and mapped in target_to_id:
                color_name = mapped
            else:
                color_name = color_from_bbox(image_bgr, cx, cy, w, h)
                report["objects_inferred_by_pixels"] += 1

            if color_name not in target_to_id:
                report["objects_dropped"] += 1
                continue

            cls_id = target_to_id[color_name]
            out_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            report["objects_by_color"][color_name] += 1

        if not out_lines:
            continue

        split = split_bucket(str(pair.image_path), args.val_ratio)
        dst_img_dir = TRAIN_IMAGES if split == "train" else VAL_IMAGES
        dst_lbl_dir = TRAIN_LABELS if split == "train" else VAL_LABELS

        stem = pair.image_path.stem
        if stem in seen_stems:
            seen_stems[stem] += 1
            stem = f"{stem}_{seen_stems[stem]}"
        else:
            seen_stems[stem] = 0

        shutil.copy2(pair.image_path, dst_img_dir / f"{stem}{pair.image_path.suffix.lower()}")
        (dst_lbl_dir / f"{stem}.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    yaml_path = write_dataset_yaml(target_names)
    report["train_images"] = len(list(TRAIN_IMAGES.glob("*")))
    report["val_images"] = len(list(VAL_IMAGES.glob("*")))
    report_path = OUT_ROOT / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Built dataset: {OUT_ROOT}")
    print(f"dataset.yaml: {yaml_path}")
    print(f"report: {report_path}")
    print(f"train/val: {report['train_images']}/{report['val_images']}")
    print(f"objects by color: {report['objects_by_color']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
