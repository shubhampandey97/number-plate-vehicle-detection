"""Score OCR on the hand-labelled crops from make_ocr_labelset.py.

Compares the ORIGINAL pipeline (Otsu, keep the single best fragment, no
format correction) against the new one (fixed-height resize, join all
fragments in reading order, Indian-format correction), across preprocessing
variants. Reports:

  exact-match accuracy  (whole plate right - what ANPR actually needs)
  CER                   (character error rate)
  read rate             (non-empty output)
  precision of output   (exact-match among plates the system *did* output -
                         a wrong plate in a log is worse than no plate)
  breakdown by lines (1 vs 2), plate height bucket, condition tag, country

The headline table uses Indian plates only (country=IN or blank), because the
format corrector is India-only and would reject Mexican CDMX plates by design.

EasyOCR on CPU: roughly 0.2-0.5 s per crop per variant. 200 crops x 5 configs
is a few minutes; run it with --variants otsu first if the CPU is busy.

Usage:
    python tools/eval_ocr.py --gt eval/ocr_gt/ocr_gt.csv --out docs/results
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.ocr import ALLOWLIST, preprocess, read_plate  # noqa: E402
from anpr.plate_text import cer, clean  # noqa: E402


def legacy_read(reader, crop):
    """Exact copy of the original repo logic (4x resize, Otsu, best fragment)."""
    img = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, score = "", 0.0
    for _, t, s in reader.readtext(th, allowlist=ALLOWLIST, paragraph=False, detail=1):
        t = re.sub(r"[^A-Z0-9]", "", t.upper())
        if s >= 0.30 and len(t) >= 4 and s > score:
            best, score = t, s
    return best


def per_plate(recs):
    """Many crops are the same car in consecutive frames. Average per unique
    plate text first, so one car seen 10 times doesn't count 10 times."""
    by = defaultdict(list)
    for r in recs:
        by[r["gt"]].append(r["pred"] == r["gt"])
    return (sum(sum(v) / len(v) for v in by.values()) / len(by)) if by else 0.0, len(by)


def h_bucket(h: int) -> str:
    return "<20px" if h < 20 else "20-40px" if h < 40 else ">=40px"


def summarise(recs):
    n = len(recs)
    if n == 0:
        return {}
    out = [r for r in recs if r["pred"]]
    return {"n": n,
            "exact": sum(r["pred"] == r["gt"] for r in recs) / n,
            "cer": sum(cer(r["pred"], r["gt"]) for r in recs) / n,
            "read_rate": len(out) / n,
            "precision": (sum(r["pred"] == r["gt"] for r in out) / len(out)) if out else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path("eval/ocr_gt/ocr_gt.csv"))
    ap.add_argument("--variants", default="otsu,gray,clahe,adaptive")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.gt))
            if clean(r["gt_text"]) and r.get("legible", "y").strip().lower() != "n"]
    if not rows:
        raise SystemExit("No labelled, legible rows in the CSV yet.")
    crops_dir = a.gt.parent / "crops"

    import easyocr
    reader = easyocr.Reader(["en"], gpu=a.gpu)

    configs = {"legacy (original repo)": lambda c: legacy_read(reader, c)}
    for v in a.variants.split(","):
        configs[f"new/{v}"] = (lambda v: lambda c: read_plate(reader, c, variant=v).text)(v)
    configs["new/otsu, no correction"] = lambda c: read_plate(reader, c, "otsu", use_correction=False).text

    results = {}
    per_crop = []
    for name, fn in configs.items():
        recs = []
        for r in rows:
            crop = cv2.imread(str(crops_dir / r["crop"]))
            pred = fn(crop) if crop is not None else ""
            rec = {"gt": clean(r["gt_text"]), "pred": pred, "lines": r.get("lines", "") or "?",
                   "country": (r.get("country", "") or "IN").strip().upper(),
                   "h": h_bucket(int(r["box_h"])), "cond": r.get("condition", "") or "untagged"}
            recs.append(rec)
            per_crop.append({"config": name, "crop": r["crop"], **rec})
        groups = defaultdict(list)
        for rec in recs:
            groups[f"lines={rec['lines']}"].append(rec)
            groups[f"height {rec['h']}"].append(rec)
            groups[f"cond={rec['cond']}"].append(rec)
            groups[f"country={rec['country']}"].append(rec)
        indian = [rec for rec in recs if rec["country"] == "IN"]
        big = [rec for rec in indian if rec["h"] != "<20px"]
        results[name] = {"all": summarise(indian) or summarise(recs), "all_countries": summarise(recs),
                         "IN, plate >=20px": summarise(big),
                         **{k: summarise(v) for k, v in sorted(groups.items())}}
        results[name]["all"]["per_plate_exact"], results[name]["all"]["n_plates"] = per_plate(indian or recs)
        s = results[name]["all"]
        print(f"{name:28s} exact {s['exact']:.3f}  CER {s['cer']:.3f}  "
              f"read {s['read_rate']:.3f}  precision {s['precision']:.3f}")

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "ocr_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_crop[0].keys()))
        w.writeheader()
        w.writerows(per_crop)
    n_in = sum((r.get("country", "") or "IN").strip().upper() == "IN" for r in rows)
    n_plates = next(iter(results.values()))["all"]["n_plates"]
    md = [f"# OCR evaluation: {n_in} legible Indian plate crops = {n_plates} unique plates "
          f"({len(rows)} legible crops in total)", "",
          "Crops come from the split-v2 test set, cut from ground-truth boxes (detector errors excluded). "
          "Many crops are the same car in consecutive frames, so `per-plate exact` averages per unique plate first.", "",
          "| config | exact match | per-plate exact | CER | read rate | precision | exact, plates >=20 px |",
          "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        s, b = res["all"], res["IN, plate >=20px"]
        md.append(f"| {name} | {s['exact']:.3f} | {s['per_plate_exact']:.3f} | {s['cer']:.3f} | "
                  f"{s['read_rate']:.3f} | {s['precision']:.3f} | "
                  + (f"{b['exact']:.3f} (n={b['n']})" if b else "-") + " |")
    best = max(results, key=lambda k: results[k]["all"]["exact"])
    md += ["", f"## Breakdown for best config: {best}", "",
           "| group | n | exact | CER |", "|---|---|---|---|"]
    for g, s in results[best].items():
        if g != "all" and s and "n" in s:
            md.append(f"| {g} | {s['n']} | {s['exact']:.3f} | {s['cer']:.3f} |")
    (a.out / "ocr_eval.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {a.out / 'ocr_eval.md'} and ocr_predictions.csv")


if __name__ == "__main__":
    main()
