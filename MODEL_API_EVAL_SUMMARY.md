# YOLO Color Model + Public API Evaluation Summary

## Scope

This report summarizes plan items:
- API shortlist + runnable benchmark
- color-first YOLO retraining
- API-vs-local comparison
- production architecture decision

## Artifacts Produced

- Dataset build script: `build_color_dataset.py`
- API benchmark script: `benchmark_public_apis.py`
- Color evaluator script: `evaluate_color_detectors.py`
- Color dataset: `data/yolo_color/`
- Trained local model: `runs/detect/color_train/weights/best.pt`
- Exported CoreML: `runs/detect/color_train/weights/best.mlpackage`

## Benchmark Results

### 1) Latency + Throughput (20 images)

Source: `runs/api_benchmark/quick_report.json`

- Local model (`runs/detect/train3/weights/best.pt`)
  - avg latency: `41.98 ms`
  - p95 latency: `42.54 ms`
  - avg predictions/image: `2.15`
- Roboflow API (`poker-chips-detector/1`)
  - avg latency: `2150.62 ms`
  - p95 latency: `3404.76 ms`
  - avg predictions/image: `2.20`

Conclusion:
- API latency is ~50x higher than local inference in this test setup.

### 2) Color Accuracy (40-image subset, IoU=0.5)

Local color model:
- report: `runs/color_eval/local_color_eval_40.json`
- macro F1: `0.6682`

Roboflow API + class remap:
- report: `runs/color_eval/roboflow_color_eval.json`
- macro F1: `0.6834`

Conclusion:
- API has only a small accuracy edge in this subset.
- Both local/API currently fail on `red` class (F1=0), indicating data/label mapping issue remains.

## API Shortlist Status

- Roboflow Serverless API: validated and runnable.
- AWS Rekognition Custom Labels: feasible, but requires model start/stop lifecycle and hosted runtime cost.
- Vertex AI AutoML endpoint: feasible, but needs deployment and cost control.

Operationally, Roboflow is the fastest path for cloud comparison; AWS/Vertex should be phase-2 optional.

## Production Architecture Decision

Recommended now:
- Primary: **local CoreML inference** (photo + realtime).
- Optional fallback: **cloud API only on low-confidence cases**.

Fallback trigger recommendation:
- if max confidence < `0.45`
- or detected tops < `1` while image quality is acceptable
- and network is available

Reasoning:
- local gives strong latency and offline capability.
- API does not currently justify always-on use due to delay and cost.

## Next Actions (Immediate)

1. Fix `red` class failure:
   - review class remap for source labels
   - add red-focused samples in difficult lighting
2. Retrain quick cycle (20 epochs) with class balance strategy.
3. Re-run `evaluate_color_detectors.py` on fixed 100+ image validation split.
4. Enable API fallback switch in iOS only after red-class regression is resolved.
