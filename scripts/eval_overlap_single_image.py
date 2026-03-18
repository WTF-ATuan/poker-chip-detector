from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Det:
    color: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def radius(self) -> float:
        return max(self.x2 - self.x1, self.y2 - self.y1) * 0.5


COLORS = ["red", "pink", "green", "black"]
LEGACY_CLASS_MAP = {
    "1": "red",
    "2": "pink",
    "3": "green",
    "4": "black",
    "5": "red",
    "10": "pink",
    "100": "black",
    "500": "green",
    "chip": "black",
}


def iou(a: Det, b: Det) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def dedup_center_radius(dets: List[Det]) -> Tuple[List[Det], int]:
    kept: List[Det] = []
    removed = 0
    for cand in sorted(dets, key=lambda d: d.conf, reverse=True):
        dup = False
        for ex in kept:
            center_dist = math.sqrt((ex.cx - cand.cx) ** 2 + (ex.cy - cand.cy) ** 2)
            bigger = max(ex.radius, cand.radius)
            smaller = max(1e-4, min(ex.radius, cand.radius))
            radius_ratio = bigger / smaller
            if center_dist < (0.60 * bigger) and radius_ratio < 1.35:
                dup = True
                break
        if dup:
            removed += 1
        else:
            kept.append(cand)
    return kept, removed


def normalize_color(name: str) -> str:
    lower = name.strip().lower()
    if lower in COLORS:
        return lower
    return LEGACY_CLASS_MAP.get(lower, lower)


def load_gt(gt_path: Path) -> Tuple[Path, List[Det]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    image_path = Path(data["image"])
    if not image_path.is_absolute():
        image_path = (gt_path.parent.parent / image_path).resolve()
    anns = data.get("annotations", [])
    gts: List[Det] = []
    for ann in anns:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        color = normalize_color(str(ann.get("color", "")))
        if color not in COLORS:
            continue
        gts.append(
            Det(
                color=color,
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                conf=1.0,
            )
        )
    return image_path, gts


def run_model(image_path: Path, model_path: Path, conf_th: float, imgsz: int) -> List[Det]:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))
    result = model.predict(
        source=str(image_path),
        conf=conf_th,
        imgsz=imgsz,
        verbose=False,
        save=False,
        max_det=300,
        device="cpu",
    )[0]
    boxes = result.boxes
    if boxes is None:
        return []
    names = model.names if hasattr(model, "names") else {}
    xyxy = boxes.xyxy.tolist()
    cls = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    out: List[Det] = []
    for b, c, s in zip(xyxy, cls, confs):
        raw = str(names.get(int(c), int(c)))
        color = normalize_color(raw)
        if color not in COLORS:
            continue
        out.append(
            Det(
                color=color,
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
                conf=float(s),
            )
        )
    return out


def match_by_class(preds: List[Det], gts: List[Det], iou_th: float) -> Dict[str, Dict[str, int]]:
    out = {c: {"tp": 0, "fp": 0, "fn": 0} for c in COLORS}
    for color in COLORS:
        p = [d for d in preds if d.color == color]
        g = [d for d in gts if d.color == color]
        used = set()
        for det in sorted(p, key=lambda d: d.conf, reverse=True):
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(g):
                if i in used:
                    continue
                score = iou(det, gt)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0 and best_iou >= iou_th:
                out[color]["tp"] += 1
                used.add(best_idx)
            else:
                out[color]["fp"] += 1
        out[color]["fn"] += (len(g) - len(used))
    return out


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def custom_score(tp: int, fp: int, fn: int, duplicates_removed: int) -> int:
    # Heuristic score requested by user: reward correct hits, penalize misses/false/duplicates.
    return (tp * 2) - (fp * 2) - (fn * 2) - duplicates_removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image overlap regression evaluator.")
    parser.add_argument("--gt", default="data/regression/image_1023_gt.json", help="GT JSON path.")
    parser.add_argument("--model", default="runs/detect/color_train/weights/best.pt", help="Model path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--iou", type=float, default=0.5, help="Matching IoU threshold.")
    parser.add_argument("--disable-dedup", action="store_true", help="Disable center/radius dedup stage.")
    parser.add_argument("--output", default="runs/regression/image_1023_eval.json", help="Output report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_path = Path(args.gt).resolve()
    image_path, gts = load_gt(gt_path)
    if not gts:
        print("GT is empty. Please annotate first with scripts/regression_label_tool.py")
        return 1

    t0 = time.time()
    raw_preds = run_model(image_path=image_path, model_path=Path(args.model).resolve(), conf_th=args.conf, imgsz=args.imgsz)
    dedup_removed = 0
    preds = raw_preds
    if not args.disable_dedup:
        preds, dedup_removed = dedup_center_radius(raw_preds)

    stats = match_by_class(preds, gts, args.iou)
    class_report = {}
    f1s = []
    total_tp = total_fp = total_fn = 0
    for color in COLORS:
        tp = stats[color]["tp"]
        fp = stats[color]["fp"]
        fn = stats[color]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p, r, f1 = prf(tp, fp, fn)
        class_report[color] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)

    overall_p, overall_r, overall_f1 = prf(total_tp, total_fp, total_fn)
    score = custom_score(total_tp, total_fp, total_fn, dedup_removed)
    report = {
        "image": str(image_path),
        "model": str(Path(args.model).resolve()),
        "iou_threshold": args.iou,
        "confidence_threshold": args.conf,
        "raw_pred_count": len(raw_preds),
        "post_dedup_pred_count": len(preds),
        "dedup_removed": dedup_removed,
        "gt_count": len(gts),
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "heuristic_score": score,
        },
        "class_report": class_report,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
