from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ColorReference:
    name: str
    bgr: tuple[int, int, int]


DEFAULT_REFERENCES = [
    ColorReference("orange", (60, 110, 225)),
    ColorReference("pink", (170, 185, 225)),
    ColorReference("green", (105, 155, 85)),
    ColorReference("purple", (170, 120, 140)),
    ColorReference("black", (55, 55, 55)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect visible chip tops and classify chip colors.")
    parser.add_argument("--input", required=True, help="Input image path or directory.")
    parser.add_argument("--output-dir", default="runs/color_pipeline_v1", help="Directory for debug output.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional max image count.")
    return parser.parse_args()


def collect_images(input_path: Path, max_images: int) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images > 0:
        return images[:max_images]
    return images


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_for_analysis(image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dim / float(max(height, width)))
    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_chip_candidates(image_bgr: np.ndarray) -> list[tuple[int, int, int, float]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=70,
        param1=130,
        param2=36,
        minRadius=24,
        maxRadius=78,
    )
    if circles is None:
        return []

    raw = np.round(circles[0, :]).astype(int)
    deduped: list[tuple[int, int, int, float]] = []
    for x, y, radius in sorted(raw, key=lambda item: item[2], reverse=True):
        keep = True
        for ex, ey, er, _ in deduped:
            center_distance = math.hypot(x - ex, y - ey)
            if center_distance < min(radius, er) * 0.8:
                keep = False
                break
        if keep:
            confidence = max(0.35, min(0.98, radius / 100.0))
            deduped.append((x, y, radius, confidence))
    return sorted(deduped, key=lambda item: (item[1], item[0]))


def circular_hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180.0 - diff)


def build_radial_masks(shape: tuple[int, int], x: int, y: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    center_mask = distance <= radius * 0.28
    ring_mask = (distance >= radius * 0.45) & (distance <= radius * 0.82)
    return center_mask, ring_mask


def is_chip_like_candidate(image_bgr: np.ndarray, x: int, y: int, radius: int) -> tuple[bool, dict[str, float]]:
    height, width = image_bgr.shape[:2]
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return False, {"reason": -1}

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    center_mask, ring_mask = build_radial_masks((height, width), x, y, radius)

    center_pixels = hsv[center_mask]
    ring_pixels = hsv[ring_mask]
    if center_pixels.size == 0 or ring_pixels.size == 0:
        return False, {"reason": -2}

    center_mean = center_pixels.mean(axis=0)
    ring_mean = ring_pixels.mean(axis=0)
    sat_delta = float(ring_mean[1] - center_mean[1])
    val_delta = float(center_mean[2] - ring_mean[2])
    ring_sat = float(ring_mean[1])
    center_val = float(center_mean[2])
    ring_val = float(ring_mean[2])
    score = ring_sat * 0.04 + sat_delta * 0.05 + center_val * 0.02 + val_delta * 0.03

    passes = (
        ring_sat > 45
        and center_val > 80
        and sat_delta > 12
        and ring_val > 60
        and score > 5.4
    )
    return passes, {
        "ring_sat": round(ring_sat, 2),
        "center_val": round(center_val, 2),
        "sat_delta": round(sat_delta, 2),
        "val_delta": round(val_delta, 2),
        "score": round(score, 2),
    }


def classify_chip_color(image_bgr: np.ndarray, x: int, y: int, radius: int) -> tuple[str, float, dict[str, float]]:
    height, width = image_bgr.shape[:2]
    _, ring_mask = build_radial_masks((height, width), x, y, radius)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    sat_mask = hsv[:, :, 1] > 35
    val_mask = hsv[:, :, 2] > 35
    combined_mask = ring_mask & sat_mask & val_mask
    if not np.any(combined_mask):
        combined_mask = ring_mask

    ring_hsv = hsv[combined_mask]
    ring_lab = lab[combined_mask]
    if ring_hsv.size == 0 or ring_lab.size == 0:
        return "unknown", 0.0, {}

    median_hsv = np.median(ring_hsv, axis=0)
    median_lab = np.median(ring_lab, axis=0)

    best_label = "unknown"
    best_score = float("inf")
    per_label_distance: dict[str, float] = {}

    for ref in DEFAULT_REFERENCES:
        ref_pixel = np.uint8([[list(ref.bgr)]])
        ref_hsv = cv2.cvtColor(ref_pixel, cv2.COLOR_BGR2HSV)[0][0].astype(float)
        ref_lab = cv2.cvtColor(ref_pixel, cv2.COLOR_BGR2LAB)[0][0].astype(float)

        hue_term = circular_hue_distance(float(median_hsv[0]), float(ref_hsv[0])) * 1.8
        sat_term = abs(float(median_hsv[1]) - float(ref_hsv[1])) * 0.08
        lab_term = float(np.linalg.norm(median_lab - ref_lab)) * 0.6
        score = hue_term + sat_term + lab_term
        per_label_distance[ref.name] = round(score, 2)
        if score < best_score:
            best_score = score
            best_label = ref.name

    confidence = max(0.2, min(0.98, 1.0 - best_score / 140.0))
    return best_label, confidence, per_label_distance


def analyze_image(image_path: Path) -> dict:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    resized_bgr, scale = resize_for_analysis(image_bgr)
    analysis_height, analysis_width = resized_bgr.shape[:2]
    candidates = detect_chip_candidates(resized_bgr)
    observations = []
    aggregated: dict[str, int] = {}

    for x, y, radius, candidate_confidence in candidates:
        passes_profile, profile_metrics = is_chip_like_candidate(resized_bgr, x, y, radius)
        if not passes_profile:
            continue
        label, color_confidence, distances = classify_chip_color(resized_bgr, x, y, radius)
        aggregated[label] = aggregated.get(label, 0) + 1
        observations.append(
            {
                "center": {"x": int(x), "y": int(y)},
                "normalized_center": {
                    "x": round(float(x) / float(analysis_width), 5),
                    "y": round(float(y) / float(analysis_height), 5),
                },
                "radius": int(radius),
                "normalized_radius": round(float(radius) / float(analysis_width), 5),
                "candidate_confidence": round(candidate_confidence, 3),
                "profile_metrics": profile_metrics,
                "predicted_color": label,
                "color_confidence": round(color_confidence, 3),
                "distance_scores": distances,
            }
        )

    return {
        "image": str(image_path),
        "analysis_scale": scale,
        "analysis_dimensions": {
            "width": analysis_width,
            "height": analysis_height,
        },
        "candidate_count": len(candidates),
        "visible_top_count_by_color": aggregated,
        "observations": observations,
    }, resized_bgr


def render_debug(image_bgr: np.ndarray, result: dict, output_path: Path) -> None:
    overlay = image_bgr.copy()
    color_map = {
        "orange": (50, 140, 255),
        "pink": (180, 170, 240),
        "green": (60, 180, 90),
        "purple": (200, 120, 170),
        "black": (40, 40, 40),
        "unknown": (220, 220, 220),
    }

    for observation in result["observations"]:
        x = observation["center"]["x"]
        y = observation["center"]["y"]
        radius = observation["radius"]
        label = observation["predicted_color"]
        confidence = observation["color_confidence"]
        color = color_map.get(label, color_map["unknown"])
        cv2.circle(overlay, (x, y), radius, color, 3)
        cv2.circle(overlay, (x, y), 3, color, -1)
        text = f"{label} {confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (max(0, x - radius), max(20, y - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    images = collect_images(input_path, args.max_images)
    if not images:
        print("No images found.")
        return 0

    manifest = []
    for image_path in images:
        try:
            result, resized_bgr = analyze_image(image_path)
            stem = image_path.stem.replace(" ", "_")
            debug_path = output_dir / f"{stem}_debug.jpg"
            json_path = output_dir / f"{stem}.json"
            render_debug(resized_bgr, result, debug_path)
            json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            manifest.append(
                {
                    "image": str(image_path),
                    "debug": str(debug_path),
                    "json": str(json_path),
                    "candidate_count": result["candidate_count"],
                    "visible_top_count_by_color": result["visible_top_count_by_color"],
                }
            )
            print(f"Analyzed {image_path.name}: {result['visible_top_count_by_color']}")
        except Exception as exc:
            manifest.append({"image": str(image_path), "error": str(exc)})
            print(f"Failed {image_path.name}: {exc}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
