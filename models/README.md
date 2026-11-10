# models/

Model weights live here and are **not committed** (`*.pt` is git-ignored).

| File | What | Where it came from |
|---|---|---|
| `yolov8l_baseline.pt` | YOLOv8l, 7 classes, 960 px, 150 epochs (May 2026) | `runs/detect/runs1/detect/traffic_object_detector-2/weights/best.pt` |

Metrics for this model are in `results/baseline_yolov8l/`. They are **validation** numbers on a split
that leaks heavily into train (see `docs/leakage/leakage_report.md`), so don't quote them as test results.
