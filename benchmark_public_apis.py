"""
Benchmark public API options for poker-chip detection.

Current executable scope:
- Roboflow Hosted API benchmark (latency + basic prediction stats)
- Local YOLO benchmark (latency + basic prediction stats)

This script is designed for shortlist validation (todo #1) and API-vs-local
comparison input (todo #4).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import uuid
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Dict, List


def read_manifest(manifest_path: Path, max_images: int) -> List[Path]:
    paths: List[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (manifest_path.parent.parent / raw).resolve()
        if p.exists():
            paths.append(p)
        if max_images > 0 and len(paths) >= max_images:
            break
    return paths


def bench_roboflow(images: List[Path], model_id: str, api_key: str, confidence: float, overlap: float) -> Dict:
    endpoint_model = model_id.strip().strip("/")
    parts = endpoint_model.split("/")
    if len(parts) >= 3:
        # serverless endpoint expects project/version
        endpoint_model = f"{parts[-2]}/{parts[-1]}"
    endpoint = f"https://serverless.roboflow.com/{endpoint_model}"
    latencies_ms: List[float] = []
    pred_count: List[int] = []
    class_counts: Dict[str, int] = {}
    errors: List[str] = []

    for image_path in images:
        try:
            query = urllib.parse.urlencode(
                {
                    "api_key": api_key,
                    "confidence": f"{confidence:.3f}",
                    "overlap": f"{overlap:.3f}",
                }
            )
            url = f"{endpoint}?{query}"
            boundary = f"----rf{uuid.uuid4().hex}"
            file_bytes = image_path.read_bytes()
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
            req = urllib.request.Request(
                url=url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = response.read().decode("utf-8")
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed)

            data = json.loads(raw)
            preds = data.get("predictions", [])
            pred_count.append(len(preds))
            for pred in preds:
                label = str(pred.get("class", "unknown"))
                class_counts[label] = class_counts.get(label, 0) + 1
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{image_path.name}: {exc}")

    return {
        "provider": "roboflow",
        "model_id": model_id,
        "images": len(images),
        "succeeded": len(latencies_ms),
        "failed": len(errors),
        "latency_ms_avg": round(mean(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p95": round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 2) if latencies_ms else None,
        "predictions_avg_per_image": round(mean(pred_count), 2) if pred_count else 0.0,
        "class_counts": class_counts,
        "errors": errors[:10],
    }


def bench_local_yolo(images: List[Path], model_path: Path, conf: float, imgsz: int) -> Dict:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))
    latencies_ms: List[float] = []
    pred_count: List[int] = []
    class_counts: Dict[str, int] = {}
    names = model.names if hasattr(model, "names") else {}

    for image_path in images:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            save=False,
            max_det=200,
            device="cpu",
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed)

        if not results:
            pred_count.append(0)
            continue

        boxes = results[0].boxes
        if boxes is None:
            pred_count.append(0)
            continue
        cls_tensor = boxes.cls
        n = int(cls_tensor.shape[0])
        pred_count.append(n)
        for cls_id in cls_tensor.tolist():
            idx = int(cls_id)
            label = str(names.get(idx, idx))
            class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "provider": "local_yolo",
        "model_path": str(model_path),
        "images": len(images),
        "succeeded": len(images),
        "failed": 0,
        "latency_ms_avg": round(mean(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p95": round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 2) if latencies_ms else None,
        "predictions_avg_per_image": round(mean(pred_count), 2) if pred_count else 0.0,
        "class_counts": class_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark public APIs and local model.")
    parser.add_argument("--manifest", default="data/validation_manifest.txt", help="Evaluation image manifest.")
    parser.add_argument("--max-images", type=int, default=30, help="Limit images for quick benchmark.")
    parser.add_argument("--output", default="runs/api_benchmark/report.json", help="Output report path.")

    parser.add_argument("--local-model", default="runs/detect/train3/weights/best.pt", help="Local YOLO model.")
    parser.add_argument("--local-conf", type=float, default=0.35, help="Local confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")

    parser.add_argument("--roboflow-api-key", default="", help="Roboflow API key.")
    parser.add_argument(
        "--roboflow-model-id",
        default="urop2023-ar-metaverse/poker-chips-detector/1",
        help="Roboflow model id: workspace/project/version.",
    )
    parser.add_argument("--roboflow-confidence", type=float, default=0.4, help="Roboflow confidence.")
    parser.add_argument("--roboflow-overlap", type=float, default=0.3, help="Roboflow NMS overlap.")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    images = read_manifest(manifest, args.max_images)
    if not images:
        print(f"No images found from manifest: {manifest}")
        return 1

    report = {
        "manifest": str(manifest),
        "images_used": len(images),
        "timestamp_unix": int(time.time()),
        "results": [],
        "notes": {
            "aws_rekognition_custom_labels": "Requires hosted model start/stop; best for phase-2 benchmark due running cost.",
            "vertex_automl_vision": "Requires endpoint deployment; include only after Roboflow baseline.",
        },
    }

    local = bench_local_yolo(images, Path(args.local_model).resolve(), args.local_conf, args.imgsz)
    report["results"].append(local)

    if args.roboflow_api_key.strip():
        rf = bench_roboflow(
            images=images,
            model_id=args.roboflow_model_id,
            api_key=args.roboflow_api_key.strip(),
            confidence=args.roboflow_confidence,
            overlap=args.roboflow_overlap,
        )
        report["results"].append(rf)
    else:
        report["results"].append(
            {
                "provider": "roboflow",
                "skipped": True,
                "reason": "No --roboflow-api-key supplied.",
                "model_id": args.roboflow_model_id,
            }
        )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark report: {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
