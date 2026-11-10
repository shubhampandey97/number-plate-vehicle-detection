# Video demo: full pipeline on British traffic footage (10 Nov 2026)

`python tools/run_video.py "data/Traffic Dataset/video11.mp4" --region UK --save-video --max-frames 300`

Clip: first 300 frames (10 s) of `video11.mp4`, 3840×2160, 30 fps, fixed camera over a queued dual carriageway.
Pipeline: YOLOv8s (split v2) @ 960 px → ByteTrack (`configs/bytetrack_anpr.yaml`) → fast-plate-ocr
(`cct-s-v2-global-model`) → UK format check → per-track vote → plate log (≥ 3 agreeing reads).
Hardware: laptop GeForce MX250 (4 GB): **2.3 fps** at 4K input.

**Caveat:** this video is one of the sources of training frames for the detector (the "Traffic Dataset"),
so it is a demo, not a held-out evaluation. The OCR model was never trained on it.

## Plate log

| plate | reads | track | frames | checked against the video |
|---|---|---|---|---|
| GX15OGJ | 137 | 21 | 0–137 | ✓ |
| MW51VSU | 66 | 18 | 0–73 | ✓ (one 1-edit variant MM51VSU merged) |
| NA13NRU | 36 | 19 | 0–35 | ✓ |
| KH05ZZK | 76 | 32 | 77–242 | ✓ |
| AP05JEO | 130 | 20 | 81–220 | ✓ |
| LM13VCV | 20 | 40 | 120–140 | ✓ |
| WR02FKD | 52 | 57 | 216–284 | ✓ |
| EY61NBG | 56 | 59 | 244–299 | ✓ |
| FJ14ZHY | 32 | 63 | 262–299 | ✓ |

**9 plates logged, 9 correct, no duplicates.** Within-track agreement 59–100 %. Plates on distant cars
(further up the queue) are too small to read and are never logged, which is the intended behaviour: the
log stays quiet rather than guessing.

Vehicle counts (once per track, majority class, track ≥ 5 frames): Car 28, Truck 3. Not hand-verified.

## How we got here (same clip)

| run | plates logged | issue found |
|---|---|---|
| original repo (EasyOCR, tracker at 640 px) | 38 entries, 2 valid UK plates | general OCR fails, no format check, single read |
| new pipeline, India-only format check | 0 | clip is British; the check correctly rejected every UK read → `--region` |
| + UK format, per-track vote | 10 (with duplicates) | Car 26 / Truck 26; AP05JEO on 4 "tracks" |
| + text-pooled log | 13 (9 real + 4 misread variants) | per-track table unchanged by tracker config |
| + tracker persist fix + per-track voting | **9, all correct** | tracker had been re-created every frame (Ultralytics binds `persist` on the first `track()` call) |
