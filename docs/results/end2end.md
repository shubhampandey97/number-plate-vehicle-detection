# End-to-end, single frame (detector -> OCR -> format check)

`models/yolov8s_v2.pt` @ 960 px, conf 0.25; readable hand-labelled test plates; plate found = IoU >= 0.5.

| subset | plates | found by detector | read correctly | logged | correct when logged |
|---|---|---|---|---|---|
| Indian | 37 (13 unique) | 1.00 | 0.14 | 7 | 0.71 |
| Indian, plate >= 20 px | 8 | 1.00 | 0.38 | 4 | 0.75 |

Single frames only. On video the pipeline votes over many frames per plate (`tools/run_video.py`).
