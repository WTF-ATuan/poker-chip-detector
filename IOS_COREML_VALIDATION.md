# iOS CoreML Integration Validation

## Model Artifacts

- PyTorch weights: `runs/detect/train3/weights/best.pt`
- CoreML package: `runs/detect/train3/weights/best.mlpackage`
- iOS bundled model: `PokerTableCompanion/PokerTableCompanion/Models/best.mlpackage`

## CoreML I/O Schema

- Input:
  - name: `image`
  - type: image
  - size: `640 x 640`
- Output:
  - name: `var_914`
  - type: `MLMultiArray`
  - shape: `[1, 13, 8400]` (`4 box channels + 9 class channels`)

## Runtime Postprocess Thresholds (iOS)

Defined in `CoreMLChipAnalyzer` and `ModelOutputMapper`:

- confidence threshold: `0.35`
- NMS IoU threshold: `0.45`
- min normalized box size: `0.01`

## Class Mapping Strategy

Implemented in `ModelOutputMapper.mapDetectionToChipConfig(...)`:

1. Exact match by class label -> `ChipColorConfig.name`
2. If class label is numeric, map to nearest `ChipColorConfig.denomination`
3. Fallback to first chip config for generic labels like `chip`

## Fixed Validation Set

- Manifest file: `data/validation_manifest.txt`
- Image count: `72`

This manifest should be reused for future regression checks to keep comparisons stable.

## Validation Results

Command:

```bash
.venv311/bin/yolo val task=detect model="runs/detect/train3/weights/best.pt" data="data/yolo/dataset.yaml" imgsz=640
```

Key metrics:

- Precision: `0.889`
- Recall: `0.938`
- mAP50: `0.949`
- mAP50-95: `0.852`

Per-class highlights:

- `chip`: mAP50-95 `0.959`
- `10`: mAP50-95 `0.709` (weakest class, needs more balanced data)
- `100`: mAP50-95 `0.868`
- `500`: mAP50-95 `0.871`

## Notes for Next Iteration

- Improve the weakest denomination class (`10`) with more labeled samples and harder lighting conditions.
- If realtime fps drops on older devices, increase `minInferenceInterval` in `RealtimeCameraAnalyzer`.
- If false positives increase in realtime mode, raise confidence threshold from `0.35` to `0.40~0.45`.
