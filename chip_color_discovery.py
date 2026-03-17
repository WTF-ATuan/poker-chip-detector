"""
Automatic poker chip color discovery.

Pipeline:
  1. HoughCircles  — detect circular chip top-faces
  2. Ring sampling — extract the 45%–75% annular band in Lab color space
  3. K-means       — cluster chip colors (k auto-selected via Silhouette Score)
  4. Report        — print how many chips belong to each color group

Usage:
    source .venv311/bin/activate
    python chip_color_discovery.py                   # all images in data/
    python chip_color_discovery.py --image data/chips\ pic1.jpg
    python chip_color_discovery.py --k 3             # force 3 color groups

Output per image:
    - Console summary of detected groups
    - Debug image saved to data/debug/<stem>_discovery.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from chip_color_utils import draw_ring_overlay, extract_ring_lab, hex_to_bgr

DATA_DIR = Path("data")
DEBUG_DIR = DATA_DIR / "debug"

# HoughCircles tuning — adjust if your photos have very different chip sizes
HOUGH_MIN_RADIUS = 15   # pixels (minimum chip top-face radius in the image)
HOUGH_MAX_RADIUS = 120  # pixels
HOUGH_PARAM1 = 80       # Canny upper threshold
HOUGH_PARAM2 = 28       # accumulator threshold (lower → more detections)

K_MIN = 2
K_MAX = 6


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ChipDetection:
    cx: float
    cy: float
    radius: float
    ring_lab: List[float]  # [L, a, b]


@dataclass
class ColorGroup:
    group_id: int
    chip_count: int
    representative_lab: List[float]
    representative_bgr: List[int]  # for display
    chip_indices: List[int]


@dataclass
class DiscoveryResult:
    image_path: str
    total_chips_detected: int
    num_color_groups: int
    color_groups: List[ColorGroup]


# ──────────────────────────────────────────────
# Step 1: HoughCircles detection
# ──────────────────────────────────────────────

def detect_chips(img_bgr: np.ndarray) -> List[ChipDetection]:
    """
    Detect circular chip top-faces using HoughCircles.
    Returns a list of ChipDetection with pixel-space (cx, cy, radius).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Light blur to reduce noise before Hough
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(HOUGH_MIN_RADIUS * 1.5),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    h, w = img_bgr.shape[:2]

    detections: List[ChipDetection] = []
    for cx, cy, r in circles:
        # Skip circles whose ring band falls mostly outside the image
        if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
            continue
        ring_lab = extract_ring_lab(img_bgr, float(cx), float(cy), float(r))
        detections.append(ChipDetection(
            cx=float(cx),
            cy=float(cy),
            radius=float(r),
            ring_lab=ring_lab.tolist(),
        ))

    return detections


# ──────────────────────────────────────────────
# Step 2: K-means color clustering
# ──────────────────────────────────────────────

def _silhouette_for_k(features: np.ndarray, k: int, seed: int = 42) -> float:
    """Return silhouette score for a given k. Returns -1 if k is invalid."""
    if k >= len(features):
        return -1.0
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(features)
    if len(set(labels)) < 2:
        return -1.0
    return float(silhouette_score(features, labels))


def auto_cluster(
    detections: List[ChipDetection],
    forced_k: Optional[int] = None,
) -> List[ColorGroup]:
    """
    Cluster detections by ring Lab color.
    If forced_k is provided, use it directly.
    Otherwise, try k in [K_MIN, K_MAX] and pick the best Silhouette Score.
    """
    if len(detections) == 0:
        return []

    features = np.array([d.ring_lab for d in detections], dtype=np.float32)

    if forced_k is not None:
        best_k = max(1, min(forced_k, len(detections)))
    elif len(detections) <= K_MIN:
        best_k = len(detections)
    else:
        k_range = range(K_MIN, min(K_MAX + 1, len(detections)))
        scores = {k: _silhouette_for_k(features, k) for k in k_range}
        best_k = max(scores, key=scores.get)
        print(f"  Silhouette scores: { {k: f'{v:.3f}' for k, v in scores.items()} }")
        print(f"  Auto-selected k = {best_k}")

    if best_k == 1:
        # Single group — no clustering needed
        mean_lab = features.mean(axis=0)
        mean_bgr = _lab_to_bgr(mean_lab)
        return [ColorGroup(
            group_id=0,
            chip_count=len(detections),
            representative_lab=mean_lab.tolist(),
            representative_bgr=mean_bgr,
            chip_indices=list(range(len(detections))),
        )]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(features)

    groups: List[ColorGroup] = []
    for group_id in range(best_k):
        indices = [i for i, label in enumerate(labels) if label == group_id]
        center_lab = km.cluster_centers_[group_id].astype(np.float32)
        groups.append(ColorGroup(
            group_id=group_id,
            chip_count=len(indices),
            representative_lab=center_lab.tolist(),
            representative_bgr=_lab_to_bgr(center_lab),
            chip_indices=indices,
        ))

    # Sort by chip count descending
    groups.sort(key=lambda g: g.chip_count, reverse=True)
    return groups


def _lab_to_bgr(lab: np.ndarray) -> List[int]:
    """Convert Lab float32 (3,) to BGR [B, G, R] list for OpenCV display."""
    lab_pixel = lab.reshape(1, 1, 3).astype(np.float32)
    bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)
    bgr_uint8 = (bgr[0, 0] * 255).clip(0, 255).astype(np.uint8)
    return bgr_uint8.tolist()


# ──────────────────────────────────────────────
# Step 3: Debug image
# ──────────────────────────────────────────────

# Distinct BGR colors for up to 6 groups (for debug overlay)
_GROUP_COLORS = [
    (0, 165, 255),   # orange
    (255, 0, 100),   # pink/magenta
    (0, 200, 0),     # green
    (180, 0, 255),   # purple
    (0, 0, 0),       # black
    (200, 200, 200), # white/grey
]


def save_debug_image(
    img_bgr: np.ndarray,
    detections: List[ChipDetection],
    groups: List[ColorGroup],
    out_path: Path,
) -> None:
    """Draw detected circles color-coded by group and save debug image."""
    out = img_bgr.copy()

    # Build chip_index → group mapping
    chip_to_group: dict[int, ColorGroup] = {}
    for g in groups:
        for idx in g.chip_indices:
            chip_to_group[idx] = g

    for idx, det in enumerate(detections):
        group = chip_to_group.get(idx)
        if group is None:
            color = (128, 128, 128)
        else:
            color = tuple(_GROUP_COLORS[group.group_id % len(_GROUP_COLORS)])

        cx, cy, r = int(det.cx), int(det.cy), int(det.radius)

        # Draw ring band bounds (faint)
        out = draw_ring_overlay(out, det.cx, det.cy, det.radius, color_bgr=color)
        # Outer circle
        cv2.circle(out, (cx, cy), r, color, 2)
        # Group label
        label = f"G{group.group_id}" if group else "?"
        cv2.putText(out, label, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Legend
    for i, g in enumerate(groups):
        legend_color = tuple(_GROUP_COLORS[g.group_id % len(_GROUP_COLORS)])
        y = 30 + i * 26
        cv2.rectangle(out, (10, y - 14), (26, y + 2), legend_color, -1)
        cv2.putText(out, f"Group {g.group_id}: {g.chip_count} chips",
                    (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, legend_color, 2)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


# ──────────────────────────────────────────────
# Main per-image function
# ──────────────────────────────────────────────

def discover_colors(
    img_path: Path,
    forced_k: Optional[int] = None,
) -> DiscoveryResult:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")

    detections = detect_chips(img_bgr)
    print(f"  Detected {len(detections)} chip candidate(s)")

    if len(detections) == 0:
        print("  No chips found — try adjusting HOUGH_PARAM2 (lower = more detections)")
        return DiscoveryResult(
            image_path=str(img_path),
            total_chips_detected=0,
            num_color_groups=0,
            color_groups=[],
        )

    groups = auto_cluster(detections, forced_k=forced_k)

    print(f"  Color groups found: {len(groups)}")
    for g in groups:
        lab = g.representative_lab
        print(f"    Group {g.group_id}: {g.chip_count} chips  "
              f"Lab=[{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]  "
              f"BGR={g.representative_bgr}")

    debug_path = DEBUG_DIR / f"{img_path.stem}_discovery.jpg"
    save_debug_image(img_bgr, detections, groups, debug_path)
    print(f"  Debug image → {debug_path}")

    return DiscoveryResult(
        image_path=str(img_path),
        total_chips_detected=len(detections),
        num_color_groups=len(groups),
        color_groups=groups,
    )


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-discover poker chip colors in images.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image (default: all data/*.jpg)")
    parser.add_argument("--k", type=int, default=None,
                        help="Force a specific number of color groups (default: auto)")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON to stdout")
    args = parser.parse_args()

    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.jpeg"))
        images = [p for p in images if "debug" not in p.stem.lower()]

    if not images:
        print("No images found. Place .jpg files in data/ or pass --image <path>.")
        return 1

    results: List[DiscoveryResult] = []
    for img_path in images:
        try:
            result = discover_colors(img_path, forced_k=args.k)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR processing {img_path}: {exc}")

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(results)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
