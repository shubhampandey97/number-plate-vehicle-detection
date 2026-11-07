"""Evaluate the detector on the held-out TEST split (never used to pick best.pt).

The 0.985 mAP@0.5 in the README comes from the val split, which Ultralytics
also used to select best.pt. This script reports test-split numbers instead.

Safety check: only test images that have a label file are evaluated (via an
explicit image list). An unlabelled image would count every correct detection
as a false positive. On the current data all 179 test images are labelled, so
this filter changes nothing; it is kept in case the folder changes.

Usage (GPU or CPU; on CPU it is slow at imgsz 960 - fine to run later):
    python tools/eval_detector_test.py --weights models/yolov8l_baseline.pt \
        --data data/final_v2/data.yaml --imgsz 960
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov8l_baseline.pt")
    ap.add_argument("--data", default="data/final_v2/data.yaml")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.data))
    root = Path(cfg.get("path", Path(a.data).parent))
    img_dir, lbl_dir = root / "images" / "test", root / "labels" / "test"
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in EXTS]
    labelled = [p for p in imgs if (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"test images: {len(imgs)}, with labels: {len(labelled)}, "
          f"unlabelled (excluded): {len(imgs) - len(labelled)}")
    if not labelled:
        raise SystemExit("No labelled test images - create a labelled test split first.")

    a.out.mkdir(parents=True, exist_ok=True)
    lst = a.out / "test_labelled.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in labelled) + "\n")
    cfg2 = dict(cfg, path=str(root.resolve()), test=str(lst.resolve()))
    tmp_yaml = a.out / "data_test_only.yaml"
    tmp_yaml.write_text(yaml.safe_dump(cfg2))

    from ultralytics import YOLO
    model = YOLO(a.weights)
    m = model.val(data=str(tmp_yaml), split="test", imgsz=a.imgsz, device=a.device,
                  plots=True, project=str(a.out), name="test_eval", exist_ok=True)
    names = model.names
    per_class = {names[c]: {"mAP50": float(m.box.ap50[i]), "mAP50_95": float(m.box.ap[i])}
                 for i, c in enumerate(m.box.ap_class_index)}
    res = {"n_images": len(labelled), "precision": float(m.box.mp), "recall": float(m.box.mr),
           "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "per_class": per_class}
    (a.out / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
