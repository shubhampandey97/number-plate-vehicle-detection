# ANPR audit (7 Nov 2026)

Scope: the `main` branch as on GitHub (`528c31b`). File paths below are from before the 8 Nov restructure (`src/ocr` → `legacy/ocr`, `src/tracking` → `legacy/tracking`, `src/analytics` → `legacy/analytics`, `streamlit_app/app.py` → `app/streamlit_app.py`, `src/dataset_engineering` → `scripts/data_prep`, `eda` → `scripts/eda`; see CLAUDE.md). Goal: find out which of the headline numbers can go on a CV as they are, and what has to be measured first.

## Summary

| Claim | Status |
|---|---|
| Detector P 0.962 / R 0.973 / mAP@0.5 0.985 (YOLOv8l, epoch 150) | **Inflated by leakage (confirmed 8 Nov). Retrained on a leak-free split: held-out test mAP50 0.707 (YOLOv8l) / 0.743 (YOLOv8s).** It's the **val** split, which also picked `best.pt`. `tools/check_leakage.py`: 4,749 cross-split near-duplicate pairs (47 byte-identical); **82.6 % of val and 86.0 % of test images** have a near-duplicate in another split. Needs a group-aware re-split and retraining before any detector number is quotable. |
| "Automatic Number Plate Recognition" | **Not measured.** OCR has never been scored. `eda/ocr_ground_truth.py` writes image names with blank text, so no labels exist. |
| End-to-end plate reading works | **Evidence says no.** `outputs/logs/plate_logs.csv` has 38 plates read by the tracker. **0 of 38** match an Indian plate format, and all are 4–8 characters, while real plates have 9–10. The format corrector (below) can't rescue any of them, so they are misreads, not small O/0-type slips. |

## Findings

### Evaluation
1. **No held-out test number.** `results.csv` metrics are on val, which was also used to choose the best checkpoint and for early stopping (`patience=40`).
2. **Split isn't reproducible.** `split_final_dataset.py` calls `random.shuffle` with no seed, so it can't be recreated.
3. **Leakage: confirmed.** "Traffic Dataset" comes from videos (`Video11.mp4`, …), and a random image-level split put consecutive frames of the same scene into train, val and test. `check_leakage.py` (dHash, Hamming ≤ 4): val 294/356 (82.6 %), test 154/179 (86.0 %) have a near-duplicate in another split; 47 cross-split pairs are byte-identical. → `tools/inspect_leakage.py` for visual confirmation and scene groups, then a group-aware split.
4. **Test split: 179 images, all labelled** (corrected 8 Nov after checking the data on disk). Earlier versions of this audit said `images/test` held unlabelled images, because of `move_unlabeled_to_test.py`, but that script reads `data/raw_datasets/images`, which doesn't exist, so it never ran on this layout. 179 = 10 % of 1,782, matching `split_final_dataset.py`. The test split was never used in training or for picking `best.pt`, so **it's a usable held-out set** (still subject to the leakage check). `eval_detector_test.py` keeps its labelled-only filter as a safety check.
5. **Mixed plate sources and unreadable "plates".** The report's batch previews (pp. 35–36) show **Mexico City (CDMX) plates**, and the same plate (A-5613-C, J95-BCS) appears in several images. The raw "Traffic Dataset" has a separate *Blur Number Plate* class (1,149 boxes, versus 479 clear ones), and `remap_and_merge_datasets.py` merges it into Number_plate. So plate detection AP mixes readable and unreadable plates from two countries. → the OCR label set records `legible` and `country`, and OCR is scored on legible Indian plates.
6. **Class imbalance:** Person has 104 train / 41 val instances, against 6,017 / 1,617 for Car. Report per-class AP and don't lean on the overall mean.

### OCR pipeline
7. **Two-line plates get truncated.** The code keeps only the single most confident EasyOCR fragment. Indian bike and auto plates are two lines (`MH12` / `AB1234`), so you get half a plate. Bike is the biggest class in the raw data (5,416 boxes). This fits the 4–8-character reads in the log. → `anpr.plate_text.join_lines` joins fragments top-to-bottom, left-to-right.
8. **No format validation or correction.** Any string of 4 or more characters is accepted and logged. → `anpr.plate_text.correct` fixes position-type confusions (O↔0, I↔1, B↔8, S↔5 …) only when the result is a valid state, RTO, series and number. Otherwise it returns nothing, because a wrong plate in a log is worse than a missing one.
9. **Fixed 4× upscale.** A 12-px plate and a 100-px plate get the same treatment. → resize to a fixed height.
10. **Same OCR code pasted 4 times** (`plate_ocr.py`, `video_ocr.py`, `vehicle_tracking.py`, Streamlit) with different settings: Otsu vs adaptive threshold, 3× vs 4×, a GPU flag. What you evaluate isn't what you deploy. → one `anpr.ocr.read_plate`.
11. **Each track is OCR'd only once,** on its first frame with conf ≥ 0.7. Often that's the smallest, farthest view of the plate. Better: vote across frames, or read when the box is largest.

### Engineering
12. **Video scripts run at `imgsz=640` while the model was trained at 960:** both `src/tracking/vehicle_tracking.py:96` and `src/analytics/traffic_analytics.py:112` (the script that wrote `plate_logs.csv`). This hurts small plates in exactly the video path.
13. **Class names are hard-coded** in 4 places. Use `model.names` instead.
14. **`streamlit_app/utils.py` loads a different, older model** (`number_plate_detector`, YOLOv8n, 1 class).
15. **`requirements.txt` couldn't be installed:** it was UTF-16 encoded, had conda `@ file:///C:/...` lines, and pinned both `opencv-python` and `-headless`. Replaced with a minimal pinned file.
16. **Hard-coded Windows paths** (`D:\Guvi\...`). `main.py` is empty, `src/detection/` and `src/utils/` are empty. No tests.
17. **The merge included the same data 2–3 times (found 8 Nov from the leakage pairs).** Files `img_000180–000934` are `.jpg` and `img_000935–001666` are `.png` copies of mostly the same frames (identical-hash pairs sit exactly 656 or 732 indices apart), and `img_000000–089` (jpg), `000090–179` (png), `000180+` (jpg) repeat the same ~90 images (`raw_datasets/dataset` and `roboflow_dataset` are the same 6-class dataset). 801 cross-split pairs have dHash distance 0. So the old val/test scores were partly computed on copies of training images. → `tools/make_group_split.py` drops copies (distance 0) and splits by scene.
18. **Source of each image is lost.** `remap_and_merge_datasets.py` renamed everything to `img_NNNNNN` and saved no mapping. The raw folders now hold about 1,236 label files (923 + 115 + 99 + 99), but the merged set has 1,782 labelled images, so the raw folders changed after the merge, or the merge used another copy (`data/Traffic Dataset` 2,166 images, `data/raw` 509, `data/dataset` 198). → Goal 5 (re-split by source) first needs **provenance recovered by matching image hashes** against every copy under `data/`. `data/raw/best.pt` is unexplained.
19. **Mixed label formats:** some label lines are polygons (`cls x1 y1 x2 y2 …`), not boxes (233 in v2 val vs 1,022 boxes). Ultralytics converts them, but any script reading `parts[1:5]` as a box gets a wrong box. → `anpr.labels.parse_line` handles both; `make_ocr_labelset.py` uses it.
20. Training itself used `seed: 0` (args.yaml); only the split script is unseeded.

## Plan

| Goal | What | CPU cost | Status |
|---|---|---|---|
| 1 | Leakage check train↔val↔test (`tools/check_leakage.py`) | light, <1 min | **done: 82.6 % val / 86.0 % test leaked** |
| 1b | Inspect pairs + scene groups (`tools/inspect_leakage.py`) | light | **done**: 588 scenes, largest 4.6 %; contact sheets show same frames / consecutive frames of fixed cameras |
| 1c | Deduplicated, group-aware, seeded split v2 (`tools/make_group_split.py` → `data/final_v2`) + re-check | light | **done**: 879/1,782 were copies → 903 unique images; train 629 / val 151 / test 123; 0 cross-split pairs |
| 1d | Retrain on split v2 on Kaggle (`notebooks/kaggle_train_v2.ipynb`): YOLOv8s and YOLOv8l @960, same hyper-parameters as the baseline; val + held-out test metrics | GPU (Kaggle) | **done**: test mAP50 0.743 (s) / 0.707 (l), see docs/results/detector_v2.md |
| 1e | Scene-level bootstrap CIs on test (local inference) | light | todo |
| 2 | OCR ground truth: ~200 GT plate crops, hand-typed (`tools/make_ocr_labelset.py`) | none (your typing time) | script ready |
| 3 | OCR evaluation, legacy vs new pipeline (`tools/eval_ocr.py`) | moderate, minutes | **first run 8 Nov**: EasyOCR fails (legacy 0/42 Indian crops, best new 1/42); see Status |
| 3b | Plate-specific OCR: (i) zero-shot fast-plate-ocr global model (`eval_ocr.py --fastplate cct-s-v2-global-model`), (ii) fine-tune on Indian plates (Kaggle), (iii) multi-frame voting in the tracker | light / GPU | (i) ready |
| 4 | Detector on labelled test split (`tools/eval_detector_test.py`) | heavy on CPU, run after TB training | script ready |
| 5 | Recover provenance (hash-match merged images to raw copies), then if leakage is found: group-aware re-split with a seed, retrain | light / GPU | todo |
| 6 | Wire `anpr.ocr` into tracker + app; multi-frame voting; tracker at 960 | light | todo |
| 7 | ONNX / FP16 export, FPS + latency benchmark CPU vs GPU | moderate | todo |
| 8 | FastAPI `/read-plate`, Docker, README results table, model card | light | todo |

## Status
- [x] Audit written, shared `src/anpr` module, 29 unit tests passing, CI workflow, requirements fixed (branch `upgrade-v2`).
- [x] CLAUDE.md added (repo map, data/weights locations, conventions)
- [x] Repo checked on disk by Claude Code (8 Nov): best.pt present (87.7 MB), test 179/179 labelled, findings 4/12/17/18 corrected
- [x] Goal 1 run (8 Nov): leakage confirmed, see docs/leakage/leakage_report.md
- [x] Repo restructured (8 Nov): app/, scripts/, legacy/, results/, reports/, models/; dead scripts and old experiment folders removed
- [x] Goal 1b (8 Nov): 588 scenes (201 singletons, 12 of 21–100 images, largest 82 = 4.6 %); pairs confirmed visually (identical frames incl. jpg/png copies; consecutive frames of the same fixed cameras); weights moved to models/
- [x] Goal 1c (8 Nov): split v2 built. **879 of 1,782 images (49 %) were duplicate copies**; 903 unique images in 588 scenes; train 629 / val 151 / test 123 (69.7/16.7/13.6 %); check_leakage on v2: 0 cross-split pairs. Boxes (train/val/test): Bike 775/151/186, Bus 197/39/50, Car 2385/439/543, Number_plate 1151/225/286, Person 56/18/24, Truck 257/49/60, Auto 552/101/137.
- [x] Goal 1d (8 Nov): Kaggle T4. **Held-out test mAP50: YOLOv8s 0.743 (mAP50-95 0.519), YOLOv8l 0.707 (0.508)** vs the reported (leaked) 0.985. Number_plate test mAP50 0.851 / 0.835. YOLOv8s chosen for deployment. docs/results/detector_v2.md
- [x] Goal 2 (8 Nov): 250 test plate crops hand-labelled (`eval/ocr_gt/ocr_gt.csv`): **only 48 (19 %) readable by a human**; 202 too small/blurred. The 48 readable crops are only **20 unique plates** (4 Delhi cars appear in 28 crops of consecutive frames); 29 of 48 are under 20 px tall.
- [x] Goal 3 first run (8 Nov, EasyOCR, GPU): legacy pipeline **0/42** Indian crops exact (CER 0.89); new pipeline best (gray) 1/42 (CER 0.87); without format check reads 93 % but CER 0.75 and 3 % precision. Under 20 px nothing is read. Conclusion: a general text reader can't read these plates; preprocessing doesn't fix it. Found and fixed: correction accepted 1-digit RTO codes outside Delhi ("HR2GEA7905" not corrected to HR26EA7905).
- [x] RTO fix re-run (8 Nov): new/gray 2/42 Indian crops exact (precision 0.29), legacy 0/42.
- [ ] Label clean-up before quoting OCR numbers: same plate labelled two ways (DL3CNN1550 / DL3CCN1550); non-Indian plates tagged IN (NU52RGY, 88184P1, 4897861, R628JPK); DL9CA06084 has 5 final digits.
- [ ] Goal 1e: bootstrap CIs
- [ ] Goal 2 labels
- [ ] Goal 3 run
- [ ] Goal 4 run
