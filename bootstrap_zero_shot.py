"""
Zero-shot bootstrap labeling for poker chips using OWL-ViT.

This script scans the `data` directory for JPG images, runs open-vocabulary
object detection with prompts related to poker chips, and writes YOLO-format
label files to `data/labels`. It also saves annotated debug images to
`data/debug` for quick visual verification.

Intended usage:
    python bootstrap_zero_shot.py

Dependencies (install in your environment):
    pip install torch torchvision transformers supervision opencv-python pillow
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import supervision as sv
from torchvision.ops import nms as torch_nms


DATA_DIR = Path("data")
OUT_LABELS_DIR = DATA_DIR / "labels"
OUT_DEBUG_DIR = DATA_DIR / "debug"


@dataclass
class DetectionConfig:
    model_name: str = "google/owlvit-base-patch32"
    text_queries: List[List[str]] = None
    score_threshold: float = 0.05
    iou_nms_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.text_queries is None:
            self.text_queries = [[
                "poker chip",
                "poker chips",
                "stack of poker chips",
                "chip stack",
                "stack of chips",
                "pile of poker chips",
                "casino chip",
                "casino chips",
                "chip pile",
            ]]


def ensure_output_dirs() -> None:
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_c = (xmin + xmax) / 2.0
    y_c = (ymin + ymax) / 2.0
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def load_model(cfg: DetectionConfig, device: str = "cpu") -> Tuple[OwlViTProcessor, OwlViTForObjectDetection, str]:
    processor = OwlViTProcessor.from_pretrained(cfg.model_name)
    model = OwlViTForObjectDetection.from_pretrained(cfg.model_name).to(device)
    return processor, model, device


def run_inference_on_image(img_path: Path, processor: OwlViTProcessor, model: OwlViTForObjectDetection, cfg: DetectionConfig, device: str = "cpu") -> sv.Detections:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    height, width = img_bgr.shape[:2]

    inputs = processor(text=cfg.text_queries, images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[height, width]], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=cfg.score_threshold
    )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")

    if boxes is None or scores is None or labels is None or len(boxes) == 0:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=float),
            confidence=np.zeros((0,), dtype=float),
            class_id=np.zeros((0,), dtype=int),
        )

    # Apply NMS using torchvision.ops.nms
    boxes_t = boxes.detach().to(device)
    scores_t = scores.detach().to(device)
    keep_idx = torch_nms(boxes_t, scores_t, cfg.iou_nms_threshold)
    boxes_kept = boxes_t[keep_idx].cpu().numpy()
    scores_kept = scores_t[keep_idx].cpu().numpy()
    labels_kept = labels.detach()[keep_idx].cpu().numpy().astype(int)

    detections = sv.Detections(
        xyxy=boxes_kept,
        confidence=scores_kept,
        class_id=labels_kept,
    )
    return detections


def write_yolo_labels(stem: str, detections: sv.Detections, img_w: int, img_h: int) -> Path:
    label_path = OUT_LABELS_DIR / f"{stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        confidences = detections.confidence
        if confidences is None:
            confidences = np.zeros((len(detections.xyxy),), dtype=float)
        for (xmin, ymin, xmax, ymax), conf in zip(detections.xyxy, confidences):
            x, y, w, h = xyxy_to_yolo(float(xmin), float(ymin), float(xmax), float(ymax), img_w=img_w, img_h=img_h)
            # single class: 0 == chip/stack
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {float(conf):.6f}\n")
    return label_path


def save_debug_image(img_bgr: np.ndarray, detections: sv.Detections, stem: str) -> Path:
    annotator = sv.BoxAnnotator()
    # Some supervision versions do not accept keyword 'labels'. Draw boxes only.
    debug_img = annotator.annotate(img_bgr.copy(), detections)
    out_path = OUT_DEBUG_DIR / f"{stem}_debug.jpg"
    cv2.imwrite(str(out_path), debug_img)
    return out_path


def process_image(img_path: Path, processor: OwlViTProcessor, model: OwlViTForObjectDetection, cfg: DetectionConfig, device: str) -> None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Skip (cannot read): {img_path}")
        return
    h, w = img_bgr.shape[:2]

    detections = run_inference_on_image(img_path, processor, model, cfg, device)
    stem = img_path.stem

    label_path = write_yolo_labels(stem, detections, img_w=w, img_h=h)
    debug_path = save_debug_image(img_bgr, detections, stem)

    print(f"Labeled: {img_path} -> {label_path} | Debug: {debug_path}")


def find_images(root: Path) -> List[Path]:
    # Only top-level JPGs under data/, ignore debug folder
    paths = [Path(p) for p in sorted(glob.glob(str(root / "*.jpg")))]
    return [p for p in paths if "debug" not in str(p).lower()]


def main() -> int:
    ensure_output_dirs()

    images = find_images(DATA_DIR)
    if not images:
        print("No images found in data/*.jpg")
        return 0

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = DetectionConfig()
    processor, model, device = load_model(cfg, device=device)

    for img_path in images:
        try:
            process_image(img_path, processor, model, cfg, device)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


