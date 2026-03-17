"""
Ring-region color extraction utilities for poker chip analysis.

Samples the annular band at 45%–75% of a chip's radius, avoiding the
center text/pattern and edge shadows/reflections. Colors are expressed
in CIE L*a*b* space for perceptually uniform distance measurement.

Intended to be imported by both chip_color_discovery.py and any future
pipeline scripts, not run directly.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Ring pixel extraction
# ──────────────────────────────────────────────

def extract_ring_pixels(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    inner_ratio: float = 0.45,
    outer_ratio: float = 0.75,
) -> np.ndarray:
    """
    Return BGR pixels that fall inside the annular region
    [inner_ratio * radius,  outer_ratio * radius] around (cx, cy).

    Returns empty (0, 3) array if no pixels qualify.
    """
    h, w = img_bgr.shape[:2]
    inner_r = inner_ratio * radius
    outer_r = outer_ratio * radius

    # Crop bounding box to avoid full-image iteration
    x0 = max(0, int(cx - outer_r))
    x1 = min(w - 1, int(cx + outer_r))
    y0 = max(0, int(cy - outer_r))
    y1 = min(h - 1, int(cy + outer_r))

    if x0 >= x1 or y0 >= y1:
        return np.empty((0, 3), dtype=np.uint8)

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    mask = (dist >= inner_r) & (dist <= outer_r)

    if not mask.any():
        return np.empty((0, 3), dtype=np.uint8)

    crop = img_bgr[y0 : y1 + 1, x0 : x1 + 1]
    return crop[mask]


def extract_ring_lab(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    inner_ratio: float = 0.45,
    outer_ratio: float = 0.75,
) -> np.ndarray:
    """
    Return the mean CIE L*a*b* color of the ring region as a float32 (3,) array.
    Returns zeros if the ring contains no pixels.
    """
    pixels = extract_ring_pixels(img_bgr, cx, cy, radius, inner_ratio, outer_ratio)
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)

    pixels_bgr = pixels.reshape(1, -1, 3).astype(np.uint8)
    pixels_lab = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2Lab)
    return pixels_lab[0].mean(axis=0).astype(np.float32)


# ──────────────────────────────────────────────
# Color space conversion helpers
# ──────────────────────────────────────────────

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """'#RRGGBB' → (B, G, R) tuple."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def hex_to_lab(hex_color: str) -> np.ndarray:
    """'#RRGGBB' → Lab float32 (3,) array."""
    b, g, r = hex_to_bgr(hex_color)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return lab[0, 0].astype(np.float32)


def lab_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Euclidean distance in Lab space (CIE76 Delta E approximation)."""
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def bgr_to_lab(bgr_pixel: Tuple[int, int, int]) -> np.ndarray:
    """Single BGR pixel → Lab float32 (3,) array."""
    b, g, r = bgr_pixel
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return lab[0, 0].astype(np.float32)


# ──────────────────────────────────────────────
# Debug helpers
# ──────────────────────────────────────────────

def draw_ring_overlay(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
    inner_ratio: float = 0.45,
    outer_ratio: float = 0.75,
) -> np.ndarray:
    """
    Draw two concentric circles representing the ring band on a copy of img_bgr.
    Useful for visually verifying that the ring hits the correct annular area.
    """
    out = img_bgr.copy()
    cv2.circle(out, (int(cx), int(cy)), int(radius * inner_ratio), color_bgr, 1)
    cv2.circle(out, (int(cx), int(cy)), int(radius * outer_ratio), color_bgr, 2)
    return out
