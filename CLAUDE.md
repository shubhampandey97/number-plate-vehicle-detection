# CLAUDE.md: AI Traffic Monitoring & ANPR

Claude Code reads this file at the start of every session. Keep it current: update the **Status** section in `docs/audit.md` and the "Current state" section below whenever a goal finishes.

## What this project is
Shubham Pandey's portfolio project. It is a GUVI major project, written up in a 48-page report, "AI Traffic Monitoring & Automatic Number Plate Recognition (ANPR) System" (PDF, not in the repo).
Pipeline: YOLOv8l detection (7 classes) → plate crop → OpenCV preprocessing → EasyOCR → ByteTrack IDs → plate logs/analytics → Streamlit dashboard (image / video / webcam).

**Purpose of the current work (branch `upgrade-v2`):** turn this into an evaluation-honest CV project, as was done for the TB repo. Measure what the report only claims, especially OCR, and fix what the measurements expose. **Never put a number on the CV or README that hasn't been measured on held-out data by a script in this repo.**

## Report claims vs. what's verified
| Report says | Reality (see `docs/audit.md`) |
|---|---|
| P 96.18 %, R 97.27 %, mAP@0.5 98.46 % | **Inflated by leakage:** val split that also chose `best.pt`; 82.6 % of val / 86.0 % of test images had a near-duplicate in train (49 % of merged data were copies). Retrained on leak-free split v2, held-out test mAP50 = **0.707 (YOLOv8l) / 0.743 (YOLOv8s)**, see docs/results/detector_v2.md. |
| "Performed OCR in real time", "Improved OCR accuracy" | OCR **never measured**. `outputs/logs/plate_logs.csv`: 0/38 reads are valid plates. |
| "Tracking improved OCR consistency significantly" | Not measured. Each track is OCR'd once, on its first frame with conf ≥ 0.7. |
| "Organizing train/validation/test splits" | Split unseeded (training itself used seed 0). Test = 179 images, all labelled, never used for training or checkpoint choice → usable held-out set. |
| Indian traffic ANPR | The "Traffic Dataset" **videos are British** (UK plates, 4K); data also contains **Mexico City (CDMX) plates** (report pp. 35–36: K02-BAJ, J95-BCS, A-5613-C; the same plate appears in several images). |
| EDA table (report p. 23): Car 4116, Number Plate 479, **Blur Number Plate 1149**, Two Wheeler 1598, Auto 1072, Bus 305, Truck 435 | These are the raw "Traffic Dataset" classes. `remap_and_merge_datasets.py` maps **both** Number Plate and Blur Number Plate → class 3. So many "plates" the detector is scored on are unreadable by design. Detection AP for plates ≠ readable plates. |

## Data and weights (NOT in git)
- Root on Shubham's PC: `D:\Guvi\Projects\major\number-plate-vehicle-detection\`
- `data/` is git-ignored:
  - `data/raw_datasets/{dataset, roboflow_dataset, Traffic Dataset, custom_dataset}`: the sources (Traffic Dataset includes videos, e.g. `Video11.mp4`).
  - `data/final_merged_datasets/{images,labels}/{train,val,test}` + `data.yaml`: the training set. Train 1,247 / val 356 / test 179 (all labelled). Files renamed `img_NNNNNN`, with **no mapping back to the source dataset**.
  - Other copies, unclear which fed the merge: `data/Traffic Dataset` (2,166 imgs, 18 videos), `data/raw` (509 imgs + an unexplained `best.pt`), `data/dataset` (198), `data/final_dataset` (115). `data/processed`, `data/samples` are empty.
- Weights (git-ignored `*.pt`): **`models/yolov8s_v2.pt`** (deploy model, split v2, test mAP50 0.743), `models/yolov8l_v2.pt` (split v2, 0.707), and **`models/yolov8l_baseline.pt`** = the old `runs/detect/runs1/detect/traffic_object_detector-2/weights/best.pt` (YOLOv8l, 960 px, 150 epochs, 87.7 MB). All scripts load this path.
- Final class ids (`remap_and_merge_datasets.py`): **0 Bike, 1 Bus, 2 Car, 3 Number_plate, 4 Person, 5 Truck, 6 Auto**. Prefer `model.names` over hard-coded lists.
- **Privacy:** plate crops and plate text are personal data. Never commit images, crops, videos or real plate logs from new runs. Labels CSV (`eval/ocr_gt/ocr_gt.csv`) is OK to commit; crops are git-ignored.

## Folder map (restructured 8 Nov 2026)
```
src/anpr/                  shared library (use this, don't copy-paste OCR code again)
  plate_text.py            clean / is_valid (Indian formats + BH) / correct (O<->0, I<->1 …, India-only)
                           / join_lines (two-line plates) / cer
  labels.py                parse_line: YOLO box OR polygon line -> (cls, cx, cy, w, h). Labels mix both formats!
  voting.py                PlateVoter: per-track votes (overlay/diagnostic) + plate_log(): plates pooled by TEXT over all
                           tracks, 1-edit variants merged (robust to tracker id swaps); edit_distance; plate->vehicle
  pipeline.py              ANPRPipeline: YOLO track (ByteTrack) -> fastplate OCR -> correct() -> voting; counts once per track.
                           THE deployed path: app, tools/run_video.py and eval_end2end use it
  ocr.py                   fastplate_reader (default, plate-trained, download retry) + EasyOCR read_plate (comparison); crop_box, preprocess(variant: gray|otsu|adaptive|clahe, fixed height 96), read_plate(reader, crop)
tools/                     evaluation CLIs (argparse, paths relative to repo root)
  check_leakage.py         dHash near-duplicates ACROSS splits -> docs/leakage/leakage_report.md + cross_split_pairs.csv
  inspect_leakage.py       contact sheets of pairs (git-ignored jpgs) + scene groups (union-find, any split) -> groups.csv
  make_group_split.py      split v2: drop duplicate copies (dHash 0), assign whole scenes (dHash<=4) to 70/15/15, seeded,
                           class-balanced -> data/final_v2 + docs/leakage/split_v2.csv/report; verifies 0 cross-split pairs
  make_ocr_labelset.py     crops GT plate boxes -> eval/ocr_gt/crops + ocr_gt.csv to hand-label
  label_ocr_app.py         Streamlit labeller: one crop at a time -> writes eval/ocr_gt/ocr_gt.csv (streamlit run tools/label_ocr_app.py)
  eval_ocr.py              legacy vs new OCR: exact match, CER, read rate, precision; by lines/height/condition/country
  run_video.py             CLI: full pipeline on a video/image -> outputs/<name>/plates.csv (plate log), plate_tracks.csv,
                           counts.json, annotated.mp4. --region IN|UK|ANY (Traffic Dataset videos are UK)
  eval_end2end.py          single-frame end-to-end: detector box -> OCR -> correct vs hand labels -> docs/results/end2end.*
  eval_detector_test.py    detector metrics on labelled test images (default weights models/yolov8l_baseline.pt)
tests/                     pytest (37 tests); CI in .github/workflows/ci.yml
app/streamlit_app.py       dashboard (Image / Video / Live) on ANPRPipeline; model selector; plates + counts tables, CSV
scripts/train.py           YOLOv8l training config (GPU, device=0), writes runs/train/ (git-ignored)
scripts/data_prep/         analysis, cleaning, conversion (COCO->YOLO), merging (remap_and_merge_datasets = the real merge),
                           splitting (split_final_dataset = the real split, UNSEEDED), validation
scripts/eda/               class frequency, image quality, lighting/weather, augmentation previews, annotations, stats
notebooks/kaggle_train_v2.ipynb  Kaggle GPU: trains YOLOv8s + YOLOv8l @960 on split v2, val+test metrics, zips results
configs/bytetrack_anpr.yaml  tracker for queued traffic (short track memory, stricter IoU): default in ANPRPipeline
results/v2_yolov8s_960/, results/v2_yolov8l_960/, results/summary_v2.md   Kaggle runs on split v2 (val + test metrics)
results/baseline_yolov8l/  args.yaml, results.csv, curves, confusion matrices, val batch previews, plate_logs_legacy.csv
reports/eda/, reports/dataset_engineering/   EDA + dataset reports (moved from eda_report/ and outputs/)
docs/                      audit.md (source of truth for progress), leakage/, results/
models/                    weights, git-ignored (models/README.md)
data/                      datasets, git-ignored
runs/, outputs/            local training runs / script outputs, git-ignored
```
Deleted on 8 Nov (still in git history): main.py (empty), streamlit_app/utils.py (old 1-class model), src/preprocessing/* (demo
scripts), src/visualization (duplicate of scripts/eda/visualize_annotations.py), eda/ocr_ground_truth.py (blank labels),
superseded data scripts (split_dataset, merge_yolo_datasets, move_unlabeled_to_test, remove_unlabeled_images, validate_dataset),
old experiment folders (runs/detect/runs, runs/detect/outputs = 1-class plate model, aborted traffic_object_detector),
train_batch*.jpg, demo jpgs in outputs/.
Run every script from the repo root.

## Environment and how to run
- Windows, VS Code + Claude Code. TB training is finished, so the machine is free. Still **ask before starting anything long** (YOLO inference over a whole split, EasyOCR over hundreds of crops, training). Long runs go in a separate PowerShell window with logging, e.g. `cmd /c "python tools\eval_ocr.py > logs\eval_ocr.log 2>&1"`.
- Env: `anpr_env` (`conda env create -f environment.yml`, Python 3.11), then torch (cu126 wheel if `nvidia-smi` works, else CPU; see `requirements.txt` header), then `pip install -r requirements.txt -r requirements-dev.txt`. Set `PYTHONNOUSERSITE=1`. The baseline was trained with `device=0` on CUDA, so this PC probably has an NVIDIA GPU: check.
- Tests: `pytest -q tests`. Lint: `ruff check src/anpr tools tests`. CI: `.github/workflows/ci.yml` runs both on push.
- The old `requirements.txt` was a UTF-16 conda export and couldn't be installed; it was replaced.

## Conventions
- New code goes in `src/anpr/` (library) or `tools/` (CLI scripts with argparse and defaults relative to the repo root). Add a test for every new function in `src/anpr`.
- Every evaluation script writes its results to `docs/results/` (markdown + csv/json) so numbers in the README can be traced to a file.
- Keep the deployed path and the evaluated path the same: tracker, app and eval must all call `anpr.ocr.read_plate`.
- Report detector metrics as "validation" until `eval_detector_test.py` has run; report per-class AP, not just the mean (Person has only 104 train instances).
- legacy/ was deleted on 10 Nov after ANPRPipeline replaced it (still in git history). The original EasyOCR method survives only as `legacy_read` in eval_ocr.py, for comparison.
- Commit messages end with the co-author line. Work on `upgrade-v2`; open a PR to `main` per finished goal.

## Current state (10 Nov 2026)
- Branch `upgrade-v2` is pushed to GitHub (backup). Not merged to main yet.
- Detector (held-out test, split v2): YOLOv8s mAP50 0.743 / YOLOv8l 0.707 (reported 0.985 was leaked val). Deploy: `models/yolov8s_v2.pt`.
- OCR on hand-labelled GT crops (37 Indian, 13 unique plates): EasyOCR legacy 0/37; best EasyOCR 2/37; **fast-plate-ocr + correction 6/37, 67 % precision** (CER raw 0.23). Correction made strict so it never invents plates from truncated reads.
- **New (10 Nov): `ANPRPipeline`** (detect + ByteTrack + fastplate + correct + multi-frame voting), used by the rewritten app and `tools/run_video.py`; legacy scripts removed. 50 tests incl. a fake-detector pipeline test.
- End-to-end single frame: detector finds 100 % of readable Indian test plates; 5/37 read, 71 % of logged plates correct. Video11 is British → use `--region UK`. Pipeline speed on MX250 at 4K: 2.2 fps.
- Video11 (UK, 300 frames): readings mostly right but ByteTrack swapped ids in the queue → tracker config + text-pooled plate log (10 Nov). Re-run to confirm.
- Next: then fine-tune fast-plate-ocr on Indian plates (Kaggle); README rewrite with honest results; FastAPI + Docker; bootstrap CIs.
