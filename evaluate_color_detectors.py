"""
Evaluate color detection models (local YOLO + optional Roboflow API) on a YOLO val split.

Outputs per-class precision/recall/F1 and macro-F1 for color classes.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Box:
    cls_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


def parse_dataset_yaml(yaml_path: Path) -> Tuple[Path, Path, List[str]]:
    text = yaml_path.read_text(encoding="utf-8")
    vals: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        vals[k.strip()] = v.strip()
    root = Path(vals.get("path", yaml_path.parent)).resolve()
    val_images = root / vals.get("val", "val/images")

    names: List[str] = []
    if "names" in vals and vals["names"].startswith("["):
        names_raw = vals["names"].strip("[]")
        names = [n.strip().strip("'\"") for n in names_raw.split(",") if n.strip()]
    else:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("- "):
                names.append(s[2:].strip().strip("'\""))
    return root, val_images, names


def read_gt_boxes(label_path: Path, image_w: int, image_h: int, class_names: List[str]) -> List[Box]:
    if not label_path.exists():
        return []
    out: List[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2.0) * image_w
        y1 = (cy - h / 2.0) * image_h
        x2 = (cx + w / 2.0) * image_w
        y2 = (cy + h / 2.0) * image_h
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        out.append(Box(cls_name=cls_name, x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0))
    return out


def iou(a: Box, b: Box) -> float:
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
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def greedy_match(preds: List[Box], gts: List[Box], iou_thr: float) -> Tuple[int, int, int]:
    used_gt = set()
    tp = 0
    fp = 0
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if idx in used_gt:
                continue
            if gt.cls_name != pred.cls_name:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            tp += 1
            used_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gts) - len(used_gt)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def predict_local(model, image_path: Path, conf: float, imgsz: int, class_names: List[str]) -> List[Box]:
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        save=False,
        max_det=200,
        device="cpu",
    )[0]
    boxes = result.boxes
    out: List[Box] = []
    if boxes is None:
        return out
    xyxy = boxes.xyxy.tolist()
    cls = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    for b, c, s in zip(xyxy, cls, confs):
        idx = int(c)
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        out.append(Box(cls_name=name, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=float(s)))
    return out


def parse_api_map(raw: str) -> Dict[str, str]:
    if not raw.strip():
        return {}
    out: Dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def predict_roboflow(
    image_path: Path,
    model_id: str,
    api_key: str,
    confidence: float,
    overlap: float,
    class_map: Dict[str, str],
) -> List[Box]:
    endpoint_model = model_id.strip().strip("/")
    parts = endpoint_model.split("/")
    if len(parts) >= 3:
        endpoint_model = f"{parts[-2]}/{parts[-1]}"
    endpoint = f"https://serverless.roboflow.com/{endpoint_model}"
    query = urllib.parse.urlencode(
        {
            "api_key": api_key,
            "confidence": f"{confidence:.3f}",
            "overlap": f"{overlap:.3f}",
        }
    )
    boundary = f"----rf{uuid.uuid4().hex}"
    file_bytes = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        url=f"{endpoint}?{query}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    preds = data.get("predictions", [])
    out: List[Box] = []
    for p in preds:
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        w = float(p.get("width", 0.0))
        h = float(p.get("height", 0.0))
        raw_name = str(p.get("class", "unknown"))
        mapped = class_map.get(raw_name, raw_name)
        out.append(
            Box(
                cls_name=mapped,
                x1=x - w / 2.0,
                y1=y - h / 2.0,
                x2=x + w / 2.0,
                y2=y + h / 2.0,
                conf=float(p.get("confidence", 0.0)),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate color detectors.")
    parser.add_argument("--dataset", default="data/yolo_color/dataset.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--provider", choices=["local", "roboflow"], default="local")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional max val images.")
    parser.add_argument("--output", default="runs/color_eval/latest.json", help="Output json report.")

    parser.add_argument("--local-model", default="runs/detect/color_train/weights/best.pt", help="Local model path.")
    parser.add_argument("--local-conf", type=float, default=0.35, help="Local confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")

    parser.add_argument("--roboflow-api-key", default="", help="Roboflow API key.")
    parser.add_argument("--roboflow-model-id", default="urop2023-ar-metaverse/poker-chips-detector/1", help="Model id.")
    parser.add_argument("--roboflow-confidence", type=float, default=0.4, help="API confidence.")
    parser.add_argument("--roboflow-overlap", type=float, default=0.3, help="API overlap.")
    parser.add_argument(
        "--roboflow-class-map",
        default="10:pink,100:black,500:green,1:red,2:pink,3:green,4:black,5:red,chip:black",
        help="Map API classes to target colors.",
    )
    args = parser.parse_args()

    root, val_images, class_names = parse_dataset_yaml(Path(args.dataset).resolve())
    image_paths = sorted(val_images.glob("*.jpg")) + sorted(val_images.glob("*.jpeg")) + sorted(val_images.glob("*.png"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print("No validation images found.")
        return 1

    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_names}
    per_image = []
    class_map = parse_api_map(args.roboflow_class_map)
    started = time.time()
    local_model = None
    if args.provider == "local":
        from ultralytics import YOLO  # type: ignore

        local_model = YOLO(str(Path(args.local_model).resolve()))

    for image_path in image_paths:
        import cv2  # type: ignore

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_gt_boxes(image_path.parent.parent / "labels" / f"{image_path.stem}.txt", w, h, class_names)

        if args.provider == "local":
            preds = predict_local(local_model, image_path, args.local_conf, args.imgsz, class_names)
        else:
            if not args.roboflow_api_key.strip():
                raise ValueError("roboflow provider selected but --roboflow-api-key is empty")
            preds = predict_roboflow(
                image_path=image_path,
                model_id=args.roboflow_model_id,
                api_key=args.roboflow_api_key.strip(),
                confidence=args.roboflow_confidence,
                overlap=args.roboflow_overlap,
                class_map=class_map,
            )

        # Compute class-wise tp/fp/fn
        img_metrics = {}
        for cls in class_names:
            cls_preds = [p for p in preds if p.cls_name == cls]
            cls_gts = [g for g in gt if g.cls_name == cls]
            tp, fp, fn = greedy_match(cls_preds, cls_gts, args.iou)
            agg[cls]["tp"] += tp
            agg[cls]["fp"] += fp
            agg[cls]["fn"] += fn
            img_metrics[cls] = {"tp": tp, "fp": fp, "fn": fn}
        per_image.append({"image": str(image_path), "metrics": img_metrics})

    class_report = {}
    f1s = []
    for cls, c in agg.items():
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        class_report[cls] = {
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)

    report = {
        "provider": args.provider,
        "dataset": str(Path(args.dataset).resolve()),
        "images": len(image_paths),
        "iou": args.iou,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "elapsed_sec": round(time.time() - started, 2),
        "class_report": class_report,
        "roboflow_class_map": class_map if args.provider == "roboflow" else {},
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
