# Single-Image Regression Workflow

Use this workflow to iterate overlap/noise algorithm changes on one fixed image.

## 1) Annotate Ground Truth (once)

```bash
source .venv311/bin/activate
python scripts/regression_label_tool.py \
  --image data/regression/image_1023.png \
  --gt data/regression/image_1023_gt.json
```

Controls:
- `1/2/3/4`: select color (`red/pink/green/black`)
- mouse drag: draw bbox
- `u`: undo
- `c`: clear
- `s`: save
- `q`: quit

## 2) Run scoring

```bash
scripts/run_regression_image_1023.sh
```

Output report:
- `runs/regression/image_1023_eval.json`

Metrics:
- Overall/class precision, recall, F1
- Macro-F1
- `raw_pred_count` vs `post_dedup_pred_count`
- `dedup_removed`
- `heuristic_score` for fast compare between algorithm versions
