# Detector results on the leak-free split v2 (8 Nov 2026)

Trained on Kaggle (Tesla T4) with `notebooks/kaggle_train_v2.ipynb`: same hyper-parameters as the original
baseline (AdamW, lr0 5e-4, cosine LR, same augmentation), 960 px, 100 epochs, `amp=True`, seed 0.
Split v2: train 629 / val 151 / test 123 images (82 test scenes); duplicate copies removed and every scene kept in
one split (`docs/leakage/split_v2_report.md`). `best.pt` was picked on **val**; **test** was used only once, at the end.

## Headline

| model | params | train time (T4) | best epoch | val mAP50 | **test P** | **test R** | **test mAP50** | **test mAP50-95** |
|---|---|---|---|---|---|---|---|---|
| YOLOv8s | 11.1 M | 48 min | 55 | 0.756 | 0.787 | 0.708 | **0.743** | **0.519** |
| YOLOv8l | 43.6 M | 159 min | 77 | 0.780 | 0.783 | 0.685 | **0.707** | **0.508** |
| *original report (leaked val, not comparable)* | 43.6 M | | 150 | *0.985* | | | | |

The original 0.985 was measured on a val split where 82.6 % of images had a near-duplicate in train, and 49 % of the
merged data were duplicate copies. On held-out scenes the same architecture scores **0.707 mAP50**, so roughly
a quarter of the reported performance came from seeing the same frames.

## Per class, test split (mAP50 / mAP50-95)

| class | test boxes | YOLOv8s | YOLOv8l |
|---|---|---|---|
| Car | 543 | 0.923 / 0.753 | 0.924 / 0.755 |
| Number_plate | 286 | 0.851 / 0.429 | 0.835 / 0.416 |
| Bike | 186 | 0.913 / 0.616 | 0.886 / 0.603 |
| Auto | 137 | 0.841 / 0.612 | 0.835 / 0.618 |
| Truck | 60 | 0.705 / 0.473 | 0.642 / 0.507 |
| Bus | 50 | 0.541 / 0.398 | 0.439 / 0.341 |
| Person | 24 | 0.429 / 0.354 | 0.387 / 0.315 |

## Reading the numbers

- **YOLOv8s is as good as YOLOv8l here** (0.743 vs 0.707 test mAP50), with 4× fewer parameters and 3.3× less
  training time. With 629 training images the large model has nothing extra to learn from. The 0.036 gap is
  probably within noise: the test set has only 123 images from 82 scenes (no confidence intervals yet).
  → **YOLOv8s is the model to deploy** (`models/yolov8s_v2.pt`).
- **Plates are found reliably but boxed loosely:** mAP50 0.85, mAP50-95 0.43. For OCR the crop is padded anyway,
  so the loose fit matters less than the miss rate. Remember the class also contains blurred, unreadable plates.
- **Weak classes have few examples:** Person (56 train / 24 test boxes) and Bus (197 / 50). Their scores are noisy
  and would need more data, not a bigger model.
- Neither run early-stopped (patience 30); best epochs 55 and 77 of 100.

## Not done yet
- Confidence intervals: scene-level bootstrap over the 123 test images (needs per-image predictions; can run locally).
- End-to-end plate reading (OCR) accuracy: needs the hand-labelled OCR set (`tools/make_ocr_labelset.py`).

Raw files: `results/v2_yolov8s_960/`, `results/v2_yolov8l_960/` (args.yaml, results.csv, curves, confusion matrices,
`metrics_v2.json` with per-class val/test), `results/summary_v2.md`.
