"""End-to-end, single-frame evaluation: detector boxes -> OCR -> format check.

eval_ocr.py scores OCR on crops cut from *ground-truth* boxes. Here the
detector has to find the plate first, and OCR runs on the *predicted* box,
exactly as in the deployed pipeline (no tracking/voting: test images are stills).

For every hand-labelled, readable plate in eval/ocr_gt/ocr_gt.csv:
  found    the detector predicted a plate box with IoU >= --iou on that plate
  correct  the reading of that predicted box (after format check) == label

Usage (repo root, anpr_env):
    python tools/eval_end2end.py --weights models/yolov8s_v2.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.labels import parse_line  # noqa: E402
from anpr.ocr import DEFAULT_FASTPLATE, crop_box, fastplate_reader  # noqa: E402
from anpr.plate_text import clean, correct  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov8s_v2.pt")
    ap.add_argument("--gt", type=Path, default=Path("eval/ocr_gt/ocr_gt.csv"))
    ap.add_argument("--data", type=Path, default=Path("data/final_v2"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.gt, newline=""))
            if clean(r["gt_text"]) and r.get("legible", "y").strip().lower() != "n"]
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["source_image"]].append(r)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    plate_cls = [k for k, v in model.names.items() if v == "Number_plate"][0]
    read = fastplate_reader(DEFAULT_FASTPLATE)

    recs = []
    for img_name, items in sorted(by_img.items()):
        img_p = a.data / "images" / "test" / img_name
        lbl_lines = (a.data / "labels" / "test" / f"{Path(img_name).stem}.txt").read_text().splitlines()
        img = cv2.imread(str(img_p))
        H, W = img.shape[:2]
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        preds = [[int(v) for v in b] for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist())
                 if c == plate_cls]
        for r in items:
            k = int(re.match(r".+_p(\d+)$", Path(r["crop"]).stem).group(1))
            _, cx, cy, w, h = parse_line(lbl_lines[k])
            gt_box = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            best = max(preds, key=lambda p: iou(p, gt_box), default=None)
            found = best is not None and iou(best, gt_box) >= a.iou
            raw = read(crop_box(img, best, pad=2)) if found else ""
            text = correct(raw).text if found else ""
            recs.append({"crop": r["crop"], "gt": clean(r["gt_text"]), "country": (r.get("country") or "IN").upper(),
                         "found": found, "raw": raw, "pred": text, "correct": text == clean(r["gt_text"]),
                         "h": int(r["box_h"])})

    def summ(rs):
        n = len(rs)
        logged = [x for x in rs if x["pred"]]
        return {"n": n, "found": sum(x["found"] for x in rs) / n if n else 0,
                "correct": sum(x["correct"] for x in rs) / n if n else 0,
                "logged": len(logged), "precision": (sum(x["correct"] for x in logged) / len(logged)) if logged else 0}

    ind = [x for x in recs if x["country"] == "IN"]
    big = [x for x in ind if x["h"] >= 20]
    res = {"weights": a.weights, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou,
           "indian": summ(ind), "indian_ge20px": summ(big), "all": summ(recs),
           "unique_indian_plates": len({x["gt"] for x in ind})}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "end2end.json").write_text(json.dumps(res, indent=2))
    with open(a.out / "end2end_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    s, b = res["indian"], res["indian_ge20px"]
    md = ["# End-to-end, single frame (detector -> OCR -> format check)", "",
          f"`{a.weights}` @ {a.imgsz} px, conf {a.conf}; readable hand-labelled test plates; plate found = IoU >= {a.iou}.", "",
          "| subset | plates | found by detector | read correctly | logged | correct when logged |", "|---|---|---|---|---|---|",
          f"| Indian | {s['n']} ({res['unique_indian_plates']} unique) | {s['found']:.2f} | {s['correct']:.2f} | {s['logged']} | {s['precision']:.2f} |",
          f"| Indian, plate >= 20 px | {b['n']} | {b['found']:.2f} | {b['correct']:.2f} | {b['logged']} | {b['precision']:.2f} |",
          "", "Single frames only. On video the pipeline votes over many frames per plate (`tools/run_video.py`)."]
    (a.out / "end2end.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
