# models/

Model weights live here and are **not committed** (`*.pt` is git-ignored).

| File | What | Held-out test mAP50 | Where it came from |
|---|---|---|---|
| `yolov8s_v2.pt` | **deployed**: YOLOv8s, 7 classes, 960 px, split v2 | **0.743** | `notebooks/kaggle_train_v2.ipynb` (Kaggle T4, 48 min) → `v2_yolov8s_960/weights/best.pt` |
| `yolov8l_v2.pt` | YOLOv8l, same data and settings | 0.707 | same notebook → `v2_yolov8l_960/weights/best.pt` |
| `yolov8l_baseline.pt` | the original model (May 2026) | not measurable* | old `runs/detect/runs1/detect/traffic_object_detector-2/weights/best.pt` |

\* The baseline was trained on a split that leaked into its validation and test sets (82.6 % / 86.0 % of
images near-duplicated in train) and on copies of most split-v2 test images, so no held-out score exists for it.
Its reported 0.985 is a leaked validation number. See `docs/audit.md`.

Metrics and curves: `results/v2_yolov8s_960/`, `results/v2_yolov8l_960/`, `results/baseline_yolov8l/`.

The plate recogniser (fast-plate-ocr `cct-s-v2-global-model`) is not stored here: it downloads itself to
`~/.cache/fast-plate-ocr/` on first use. If GitHub times out, `src/anpr/ocr.py` prints the URL and target path
for a manual download.
