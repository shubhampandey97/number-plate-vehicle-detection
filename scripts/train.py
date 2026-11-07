"""Train YOLOv8 on the leak-free split v2.

Defaults are set for the local GPU (GeForce MX250, 4 GB, Pascal):
  * YOLOv8s at 960 px: YOLOv8l at 960 needs >4 GB and ~1.5 days per 100 epochs here.
  * amp=False: Pascal consumer GPUs run FP16 very slowly, so AMP would slow training down.
  * batch 4, workers 2, no RAM cache.
Hyper-parameters otherwise follow the original baseline run
(results/baseline_yolov8l/args.yaml): AdamW, lr0 5e-4, cosine LR, same augmentation.

Usage (repo root, anpr_env):
    python scripts/train.py --smoke          # ~2 min: 1 epoch on 10% of data, checks memory + speed
    python scripts/train.py                  # full run -> runs/train/v2_yolov8s_960/
    python scripts/train.py --model yolov8l.pt --imgsz 960 --batch 8 --amp   # on a bigger GPU (Kaggle/Colab)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def absolute_data_yaml(data: Path, out_dir: Path) -> Path:
    """Write a copy of data.yaml with an absolute `path`, so Ultralytics doesn't
    resolve a relative path against its own datasets directory."""
    cfg = yaml.safe_load(open(data))
    root = Path(cfg.get("path") or data.parent)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    cfg["path"] = str(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{data.parent.name}_abs.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/final_v2/data.yaml"))
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (only on GPUs with fast FP16)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of the data to test speed/memory")
    a = ap.parse_args()

    from ultralytics import YOLO

    data = absolute_data_yaml(a.data, Path("runs/train"))
    name = a.name or f"v2_{Path(a.model).stem}_{a.imgsz}" + ("_smoke" if a.smoke else "")
    model = YOLO(a.model)
    model.train(
        data=str(data), epochs=1 if a.smoke else a.epochs, fraction=0.1 if a.smoke else 1.0,
        imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers, patience=a.patience,
        amp=a.amp, cache=False, seed=0, deterministic=True, pretrained=True,
        optimizer="AdamW", lr0=0.0005, lrf=0.01, weight_decay=0.0005, momentum=0.937, cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0005, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.1,
        close_mosaic=10, val=True, save=True, save_period=-1, plots=True, verbose=True,
        project="runs/train", name=name, exist_ok=a.smoke,
    )


if __name__ == "__main__":
    main()
