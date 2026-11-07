# CLAUDE.md: AI Traffic Monitoring & ANPR

Claude Code reads this file at the start of every session. Keep it current: update the **Status** section in `docs/audit.md` and the "Current state" section below whenever a goal finishes.

## What this project is
Shubham Pandey's portfolio project. It is a GUVI major project, written up in a 48-page report, "AI Traffic Monitoring & Automatic Number Plate Recognition (ANPR) System" (PDF, not in the repo).
Pipeline: YOLOv8l detection (7 classes) → plate crop → OpenCV preprocessing → EasyOCR → ByteTrack IDs → plate logs/analytics → Streamlit dashboard (image / video / webcam).

**Purpose of the current work (branch `upgrade-v2`):** turn this into an evaluation-honest CV project, as was done for the TB repo. Measure what the report only claims, especially OCR, and fix what the measurements expose. **Never put a number on the CV or README that hasn't been measured on held-out data by a script in this repo.**

## Report claims vs. what's verified
| Report says | Reality (see `docs/audit.md`) |
|---|---|
| P 96.18 %, R 97.27 %, mAP@0.5 98.46 % | Real, but on the **val** split, which also chose `best.pt`. No test number yet. |
| "Performed OCR in real time", "Improved OCR accuracy" | OCR **never measured**. `outputs/logs/plate_logs.csv`: 0/38 reads are valid plates. |
| "Tracking improved OCR consistency significantly" | Not measured. Each track is OCR'd once, on its first frame with conf ≥ 0.7. |
| "Organizing train/validation/test splits" | Split unseeded. `images/test` also contains **unlabelled** images. |
| Indian traffic ANPR | Data also contains **Mexico City (CDMX) plates** (report pp. 35–36: K02-BAJ, J95-BCS, A-5613-C; the same plate appears in several images). |
| EDA table (report p. 23): Car 4116, Number Plate 479, **Blur Number Plate 1149**, Two Wheeler 1598, Auto 1072, Bus 305, Truck 435 | These are the raw "Traffic Dataset" classes. `remap_and_merge_datasets.py` maps **both** Number Plate and Blur Number Plate → class 3. So many "plates" the detector is scored on are unreadable by design. Detection AP for plates ≠ readable plates. |

## Data and weights (NOT in git)
- Root on Shubham's PC: `D:\Guvi\Projects\major\number-plate-vehicle-detection\`
- `data/` is git-ignored:
  - `data/raw_datasets/{dataset, roboflow_dataset, Traffic Dataset, custom_dataset}`: the sources (Traffic Dataset includes videos, e.g. `Video11.mp4`).
  - `data/final_merged_datasets/{images,labels}/{train,val,test}` + `data.yaml`: the training set. Train 1,247 / val 356 images. Test = 10 % labelled + extra unlabelled images.
- Weights (git-ignored `*.pt`): **`runs/detect/runs1/detect/traffic_object_detector-2/weights/best.pt`** (YOLOv8l, 960 px, 150 epochs). Older runs in `runs/detect/runs/...` (YOLOv8n, stopped after 6 epochs) and `runs/detect/outputs/training/number_plate_detector` (1-class plate model, still used by `streamlit_app/utils.py`).
- Final class ids (`remap_and_merge_datasets.py`): **0 Bike, 1 Bus, 2 Car, 3 Number_plate, 4 Person, 5 Truck, 6 Auto**. Prefer `model.names` over hard-coded lists.
- **Privacy:** plate crops and plate text are personal data. Never commit images, crops, videos or real plate logs from new runs. Labels CSV (`eval/ocr_gt/ocr_gt.csv`) is OK to commit; crops are git-ignored.

## Folder map
```
src/anpr/                  NEW shared library (use this, don't copy-paste OCR code again)
  plate_text.py            clean / is_valid (Indian formats + BH) / correct (O<->0, I<->1 …, India-only)
                           / join_lines (two-line plates) / cer
  ocr.py                   crop_box, preprocess(variant: gray|otsu|adaptive|clahe, fixed height 96), read_plate(reader, crop)
tools/                     NEW evaluation scripts (CLI, argparse, no hard-coded paths)
  check_leakage.py         dHash near-duplicate search across train/val/test (CPU-light)
  make_ocr_labelset.py     crops GT plate boxes → eval/ocr_gt/crops + ocr_gt.csv to hand-label
  eval_ocr.py              legacy vs new OCR: exact match, CER, read rate, precision; by lines/height/condition/country
  eval_detector_test.py    detector metrics on LABELLED test images only (heavy on CPU)
tests/                     pytest (29 tests): plate_text, ocr (fake reader), leakage
docs/audit.md              findings + goal plan + Status checklist  ← source of truth for progress

src/training/train.py      YOLOv8l training config (GPU, device=0)
src/inference/             predict.py (batch images), predict_and_crop.py (save crops)
src/ocr/                   plate_ocr.py (single image), video_ocr.py (video, OCR every 5 frames)   [legacy, duplicated OCR]
src/tracking/vehicle_tracking.py   ByteTrack + OCR once per track, imgsz=640 (!)                 [legacy]
src/analytics/traffic_analytics.py tracking + counts + writes outputs/logs/plate_logs.csv         [legacy]
src/preprocessing/         enhancement/normalization/augmentation demos, ocr_preprocessing.py (adaptive threshold variant)
src/visualization/         annotation drawing
src/dataset_engineering/   conversion (COCO→YOLO), merging (remap_and_merge = the real one), cleaning,
                           splitting (split_final_dataset = the real one, UNSEEDED), validation, analysis
eda/ + eda_report/         class frequency, image quality, lighting/weather, augmentation previews, stats
streamlit_app/app.py       dashboard (Image / Video / Live modes), loads traffic_object_detector-2
streamlit_app/utils.py     loads the OLD 1-class plate model, different preprocessing
runs/detect/runs1/detect/traffic_object_detector-2/   results.csv, curves, confusion matrix, batch previews
outputs/                   plate_logs.csv (old run), dataset engineering reports
main.py                    empty
```
Legacy scripts load models at import time, use Windows `\` relative paths and must be run from the repo root.

## Environment and how to run
- Windows, VS Code + Claude Code. The PC is CPU-only right now, and the TB project may be training in another window. **Ask before starting anything heavy** (YOLO inference over a whole split, EasyOCR over hundreds of crops, training). Long runs go in a separate PowerShell window with logging, e.g. `cmd /c "python tools\eval_ocr.py > logs\eval_ocr.log 2>&1"`.
- Don't reuse the TB conda env (TensorFlow). Use a separate env for this repo, e.g. `anpr_env`: install torch (CPU or cu126 wheel, see the `requirements.txt` header), then `pip install -r requirements.txt -r requirements-dev.txt`. Set `PYTHONNOUSERSITE=1`.
- Tests: `pytest -q tests`. Lint: `ruff check src/anpr tools tests`. CI: `.github/workflows/ci.yml` runs both on push.
- The old `requirements.txt` was a UTF-16 conda export and couldn't be installed; it was replaced.

## Conventions
- New code goes in `src/anpr/` (library) or `tools/` (CLI scripts with argparse and defaults relative to the repo root). Add a test for every new function in `src/anpr`.
- Every evaluation script writes its results to `docs/results/` (markdown + csv/json) so numbers in the README can be traced to a file.
- Keep the deployed path and the evaluated path the same: tracker, app and eval must all call `anpr.ocr.read_plate`.
- Report detector metrics as "validation" until `eval_detector_test.py` has run; report per-class AP, not just the mean (Person has only 104 train instances).
- Don't rewrite or delete the legacy scripts until the replacement is wired in and tested. Then remove the duplicates in one commit.
- Commit messages end with the co-author line. Work on `upgrade-v2`; open a PR to `main` per finished goal.

## Current state (7 Nov 2026)
- Done: audit, `src/anpr`, tools, 29 tests, CI, requirements fixed (commit `1f58ed0`, then the CLAUDE.md/country commit).
- Next for Shubham (CPU-light): `python tools/check_leakage.py`, then `python tools/make_ocr_labelset.py --split test`, then hand-label ~200 crops (gt_text, legible, lines, condition, country).
- Then: `eval_ocr.py` → `eval_detector_test.py` (after TB training) → re-split by source if there is leakage → wire `anpr.ocr` into the tracker and app, add multi-frame voting, run the tracker at 960 → ONNX/FP16 benchmark → FastAPI + Docker → README results table + model card.
