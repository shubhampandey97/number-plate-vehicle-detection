"""Build an OCR ground-truth set from GROUND-TRUTH plate boxes (not detector
boxes), so OCR is scored on its own, independent of detection errors.

For every Number_plate box in the YOLO labels of the chosen split, save the
crop and add a row to a CSV with an empty `gt_text` column. You then type the
plate text by hand (open the crop, read it, type it). ~150-200 plates is
enough for a first, honest OCR number. No model is run: CPU-light.

Columns you fill in:
  gt_text   plate text, A-Z0-9 only, no spaces (e.g. MH12AB1234)
  legible   y / n  (n = a human cannot read it either; excluded from accuracy)
  lines     1 / 2  (two-line plates are common on bikes and autos)
  condition free text tag, e.g. day, night, blur, angle, small
  country   IN / MX / other  (the merged data also contains Mexico City
            "CDMX" plates; the plate-format corrector is India-only)

Usage:
    python tools/make_ocr_labelset.py --root data/final_merged_datasets \
        --split test --out eval/ocr_gt --max 250
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.labels import parse_line  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png")


def find_image(img_dir: Path, stem: str):
    for e in EXTS:
        p = img_dir / f"{stem}{e}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/final_v2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--plate-class", type=int, default=3, help="Number_plate class id")
    ap.add_argument("--out", type=Path, default=Path("eval/ocr_gt"))
    ap.add_argument("--max", type=int, default=250)
    ap.add_argument("--min-h", type=int, default=8, help="skip crops shorter than this (px)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    img_dir, lbl_dir = a.root / "images" / a.split, a.root / "labels" / a.split
    out_csv = a.out / "ocr_gt.csv"
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists - not overwriting your labels.")
    crops_dir = a.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        for k, line in enumerate(lbl.read_text().splitlines()):
            box = parse_line(line)          # handles box and polygon lines
            if box and box[0] == a.plate_class:
                candidates.append((lbl.stem, k, *box[1:]))
    random.Random(a.seed).shuffle(candidates)

    rows = []
    for stem, k, cx, cy, w, h in candidates:
        if len(rows) >= a.max:
            break
        p = find_image(img_dir, stem)
        if p is None:
            continue
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if y2 - y1 < a.min_h or x2 <= x1:
            continue
        name = f"{stem}_p{k}.png"
        cv2.imwrite(str(crops_dir / name), img[y1:y2, x1:x2])
        rows.append({"crop": name, "source_image": p.name, "box_w": x2 - x1, "box_h": y2 - y1,
                     "gt_text": "", "legible": "", "lines": "", "condition": "", "country": ""})

    if not rows:
        raise SystemExit(f"No Number_plate boxes found in {lbl_dir}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(candidates)} plate boxes in {a.split}; wrote {len(rows)} crops to {crops_dir}")
    print(f"Now fill gt_text / legible / lines / condition / country in {out_csv}")


if __name__ == "__main__":
    main()
