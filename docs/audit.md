# ANPR audit (7 Nov 2026)

Scope: the `main` branch as on GitHub (`528c31b`). Goal: find out which of the headline numbers can go on a CV as they are, and what has to be measured first.

## Summary

| Claim | Status |
|---|---|
| Detector P 0.962 / R 0.973 / mAP@0.5 0.985 (YOLOv8l, epoch 150) | **Real but optimistic.** It's the **val** split, and the same split picked `best.pt`. The split was random, with no seed, over data that includes video frames, so near-duplicate frames may sit in both train and val. Needs a test-split number and a leakage check. |
| "Automatic Number Plate Recognition" | **Not measured.** OCR has never been scored. `eda/ocr_ground_truth.py` writes image names with blank text, so no labels exist. |
| End-to-end plate reading works | **Evidence says no.** `outputs/logs/plate_logs.csv` has 38 plates read by the tracker. **0 of 38** match an Indian plate format, and all are 4–8 characters, while real plates have 9–10. The format corrector (below) can't rescue any of them, so they are misreads, not small O/0-type slips. |

## Findings

### Evaluation
1. **No held-out test number.** `results.csv` metrics are on val, which was also used to choose the best checkpoint and for early stopping (`patience=40`).
2. **Split isn't reproducible.** `split_final_dataset.py` calls `random.shuffle` with no seed, so it can't be recreated.
3. **Possible leakage.** "Traffic Dataset" comes from videos (`Video11.mp4`, …). A random image-level split puts consecutive frames of the same car and plate into both train and val. → `tools/check_leakage.py`.
4. **Test folder holds unlabelled images.** `move_unlabeled_to_test.py` moved images *without labels* into `images/test`. If you run `model.val(split="test")` directly, every correct detection on those images counts as a false positive. → `tools/eval_detector_test.py` evaluates only labelled test images.
5. **Class imbalance:** Person has 104 train / 41 val instances, against 6,017 / 1,617 for Car. Report per-class AP and don't lean on the overall mean.

### OCR pipeline
6. **Two-line plates get truncated.** The code keeps only the single most confident EasyOCR fragment. Indian bike and auto plates are two lines (`MH12` / `AB1234`), so you get half a plate. Bike is the biggest class in the raw data (5,416 boxes). This fits the 4–8-character reads in the log. → `anpr.plate_text.join_lines` joins fragments top-to-bottom, left-to-right.
7. **No format validation or correction.** Any string of 4 or more characters is accepted and logged. → `anpr.plate_text.correct` fixes position-type confusions (O↔0, I↔1, B↔8, S↔5 …) only when the result is a valid state, RTO, series and number. Otherwise it returns nothing, because a wrong plate in a log is worse than a missing one.
8. **Fixed 4× upscale.** A 12-px plate and a 100-px plate get the same treatment. → resize to a fixed height.
9. **Same OCR code pasted 4 times** (`plate_ocr.py`, `video_ocr.py`, `vehicle_tracking.py`, Streamlit) with different settings: Otsu vs adaptive threshold, 3× vs 4×, a GPU flag. What you evaluate isn't what you deploy. → one `anpr.ocr.read_plate`.
10. **Each track is OCR'd only once,** on its first frame with conf ≥ 0.7. Often that's the smallest, farthest view of the plate. Better: vote across frames, or read when the box is largest.

### Engineering
11. **The tracker runs at `imgsz=640` while the model was trained and is predicted at 960,** which hurts small plates in exactly the video path.
12. **Class names are hard-coded** in 4 places. Use `model.names` instead.
13. **`streamlit_app/utils.py` loads a different, older model** (`number_plate_detector`, YOLOv8n, 1 class).
14. **`requirements.txt` couldn't be installed:** it was UTF-16 encoded, had conda `@ file:///C:/...` lines, and pinned both `opencv-python` and `-headless`. Replaced with a minimal pinned file.
15. **Hard-coded Windows paths** (`D:\Guvi\...`). `main.py` is empty. No tests.

## Plan

| Goal | What | CPU cost | Status |
|---|---|---|---|
| 1 | Leakage check train↔val↔test (`tools/check_leakage.py`) | light, <1 min | script ready, run it |
| 2 | OCR ground truth: ~200 GT plate crops, hand-typed (`tools/make_ocr_labelset.py`) | none (your typing time) | script ready |
| 3 | OCR evaluation, legacy vs new pipeline (`tools/eval_ocr.py`) | moderate, minutes | script ready (needs goal 2) |
| 4 | Detector on labelled test split (`tools/eval_detector_test.py`) | heavy on CPU, run after TB training | script ready |
| 5 | If leakage is found: group-aware re-split (by source video / dataset) with a seed, then retrain | GPU | todo |
| 6 | Wire `anpr.ocr` into tracker + app; multi-frame voting; tracker at 960 | light | todo |
| 7 | ONNX / FP16 export, FPS + latency benchmark CPU vs GPU | moderate | todo |
| 8 | FastAPI `/read-plate`, Docker, README results table, model card | light | todo |

## Status
- [x] Audit written, shared `src/anpr` module, 29 unit tests passing, CI workflow, requirements fixed (branch `upgrade-v2`).
- [ ] Goal 1 run
- [ ] Goal 2 labels
- [ ] Goal 3 run
- [ ] Goal 4 run
