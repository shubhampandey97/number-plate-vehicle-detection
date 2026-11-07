# legacy/

The original pipeline scripts, kept so the old behaviour can still be run and compared.
They are being replaced by `src/anpr/` (one shared OCR path, Indian plate checks, multi-frame voting).
Once a replacement is wired in and tested, the matching script here is deleted.

| Script | What it does | Known problems (see docs/audit.md) |
|---|---|---|
| `ocr/plate_ocr.py` | detect + OCR one image | keeps only one OCR fragment (two-line plates cut in half) |
| `ocr/video_ocr.py` | detect + OCR a video, OCR every 5 frames | same OCR issue |
| `tracking/vehicle_tracking.py` | ByteTrack + OCR once per track | imgsz 640 (model trained at 960) |
| `analytics/traffic_analytics.py` | tracking + counts, wrote `plate_logs_legacy.csv` | imgsz 640; 0/38 logged plates valid |
| `inference/predict.py`, `predict_and_crop.py` | batch detection on `inference/input/` | none major |

All load `models/yolov8l_baseline.pt` and must be run from the repo root, e.g.
`python legacy/ocr/plate_ocr.py`. Several still contain hard-coded input paths at the bottom.
