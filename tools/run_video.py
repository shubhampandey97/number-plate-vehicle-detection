"""Run the full ANPR pipeline on a video (or an image) from the command line.

Writes to --out (default outputs/<video name>/):
  plates.csv        the plate log: one row per distinct plate text, pooled over
                    all tracks (robust to tracker id swaps), >= --min-reads reads
  plate_tracks.csv  diagnostic: voted text per plate track, incl. unread tracks
  counts.json       vehicles counted once per track
  annotated.mp4     (with --save-video) boxes, track ids, voted plate text

Usage (repo root, anpr_env):
    python tools/run_video.py path/to/video.mp4 --save-video
    python tools/run_video.py "data/Traffic Dataset/video11.mp4" --region UK --save-video   # British footage
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


TRACK_COLS = ["plate", "agreeing_reads", "valid_reads", "ocr_reads", "vote_share", "first_frame",
              "last_frame", "plate_track", "vehicle_track", "vehicle"]
LOG_COLS = ["plate", "reads", "vehicle", "first_frame", "last_frame", "tracks", "merged_variants"]


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
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
    ap.add_argument("--min-votes", type=int, default=2, help="agreeing valid reads per plate track (overlay)")
    ap.add_argument("--min-reads", type=int, default=3, help="agreeing valid reads needed in the plate log")
    ap.add_argument("--log-by", default="track", choices=["track", "reads"],
                    help="track: each track votes first (default); reads: pool every read (if tracking is poor)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--region", default="IN", choices=["IN", "UK", "ANY"],
                    help="plate format rules; the 'Traffic Dataset' videos are British (UK)")
    ap.add_argument("--min-track-frames", type=int, default=5,
                    help="count a vehicle only if its track lasts this many frames")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = a.out or Path("outputs") / a.source.stem
    out.mkdir(parents=True, exist_ok=True)
    pipe = ANPRPipeline(a.weights, imgsz=a.imgsz, conf=a.conf, device=a.device,
                        ocr_every=a.ocr_every, min_votes=a.min_votes, region=a.region)

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
                      f"plates so far: {len(pipe.plates(min_reads=a.min_reads))}", flush=True)
        cap.release()
        if writer:
            writer.release()
        secs = time.time() - t0

    plates = pipe.plates(min_reads=1 if frames == 1 else a.min_reads, by=a.log_by)
    by_track = pipe.plates_by_track(include_unread=True)
    write_csv(out / "plates.csv", plates, LOG_COLS)
    write_csv(out / "plate_tracks.csv", by_track, TRACK_COLS)
    counts = pipe.vehicle_counts(min_frames=1 if frames == 1 else a.min_track_frames)
    summary = {"source": str(a.source), "frames": frames, "seconds": round(secs, 1),
               "fps": round(frames / secs, 2) if secs else None, "weights": a.weights, "imgsz": a.imgsz,
               "region": a.region, "min_votes": a.min_votes, "ocr_every": a.ocr_every,
               "min_track_frames": a.min_track_frames, "vehicle_counts": counts,
               "min_reads": a.min_reads, "log_by": a.log_by, "plate_tracks": len(by_track),
               "plates_logged": len(plates)}
    (out / "counts.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for p in plates:
        merged = f'  (merged: {p["merged_variants"]})' if p["merged_variants"] else ""
        print(f'  {p["plate"]:12s} {p["reads"]:4d} reads  frames {p["first_frame"]}-{p["last_frame"]}  '
              f'tracks [{p["tracks"]}]  {p["vehicle"] or ""}{merged}')
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
