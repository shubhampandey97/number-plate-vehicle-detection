"""Run the full ANPR pipeline on a video (or an image) from the command line.

Writes to --out (default outputs/<video name>/):
  plates.csv        one row per plate track that reached a voted reading
  plates_all.csv    every plate track, including ones never read reliably
  counts.json       vehicles counted once per track
  annotated.mp4     (with --save-video) boxes, track ids, voted plate text

Usage (repo root, anpr_env):
    python tools/run_video.py path/to/video.mp4 --save-video
    python tools/run_video.py path/to/image.jpg
    python tools/run_video.py video.mp4 --ocr-every 2 --max-frames 600   # faster on the MX250
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.pipeline import ANPRPipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def write_csv(path, rows):
    cols = ["plate", "agreeing_reads", "valid_reads", "ocr_reads", "vote_share", "first_frame",
            "last_frame", "plate_track", "vehicle_track", "vehicle"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--weights", default="models/yolov8s_v2.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
    ap.add_argument("--ocr-every", type=int, default=1, help="read plates every N frames")
    ap.add_argument("--min-votes", type=int, default=2, help="agreeing valid reads needed per plate")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = a.out or Path("outputs") / a.source.stem
    out.mkdir(parents=True, exist_ok=True)
    pipe = ANPRPipeline(a.weights, imgsz=a.imgsz, conf=a.conf, device=a.device,
                        ocr_every=a.ocr_every, min_votes=a.min_votes)

    if a.source.suffix.lower() in IMG_EXTS:
        pipe.reset(min_votes=1)
        img = cv2.imread(str(a.source))
        cv2.imwrite(str(out / "annotated.jpg"), pipe.process(img, track=False))
        frames, secs = 1, 0.0
    else:
        cap = cv2.VideoCapture(str(a.source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        pipe.reset()
        frames, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok or (a.max_frames and frames >= a.max_frames):
                break
            vis = pipe.process(frame, track=True, draw=a.save_video)
            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)
            frames += 1
            if frames % 50 == 0:
                print(f"frame {frames}/{n_total}  {frames / (time.time() - t0):.1f} fps  "
                      f"plates so far: {len(pipe.plates())}", flush=True)
        cap.release()
        if writer:
            writer.release()
        secs = time.time() - t0

    plates, all_plates = pipe.plates(), pipe.plates(include_unread=True)
    write_csv(out / "plates.csv", plates)
    write_csv(out / "plates_all.csv", all_plates)
    counts = pipe.vehicle_counts()
    summary = {"source": str(a.source), "frames": frames, "seconds": round(secs, 1),
               "fps": round(frames / secs, 2) if secs else None, "weights": a.weights, "imgsz": a.imgsz,
               "min_votes": a.min_votes, "ocr_every": a.ocr_every, "vehicle_counts": counts,
               "plate_tracks": len(all_plates), "plates_read": len(plates)}
    (out / "counts.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for p in plates:
        print(f'  {p["plate"]:12s} {p["agreeing_reads"]}/{p["valid_reads"]} agreeing valid reads '
              f'({p["ocr_reads"]} attempts)  {p["vehicle"] or ""}')
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
