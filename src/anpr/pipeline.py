"""End-to-end ANPR pipeline: detect + track + read plates + vote per track.

Replaces the legacy scripts and the old app logic, which
* ran the detector at 640 px although it was trained at 960,
* counted vehicles per frame (a car seen in 100 frames was counted 100 times),
* read each plate once with EasyOCR and logged whatever came out.

Here:
* YOLOv8s trained on the leak-free split v2 (`models/yolov8s_v2.pt`), 960 px,
* ByteTrack IDs: vehicles are counted once per track,
* every plate track is read in every frame (or every `ocr_every` frames) with
  the plate-trained reader, each read is checked against Indian plate formats,
  and the plate is reported only when >= `min_votes` valid reads agree.

Usage:
    pipe = ANPRPipeline()                  # loads models
    pipe.reset()                           # new video
    for frame in frames:
        annotated = pipe.process(frame)    # BGR in, annotated BGR out
    plates = pipe.plates()                 # list of dicts (one per plate track)
    counts = pipe.vehicle_counts()         # {"Car": 12, ...}
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .ocr import DEFAULT_FASTPLATE, crop_box, fastplate_reader
from .plate_text import correct
from .voting import PlateVoter, assign_plate_to_vehicle

PLATE_CLASS = "Number_plate"
COLORS = {"Number_plate": (0, 0, 255), "Car": (0, 200, 0), "Bike": (255, 160, 0), "Auto": (0, 200, 255),
          "Bus": (200, 0, 200), "Truck": (120, 60, 0), "Person": (200, 200, 200)}


class ANPRPipeline:
    def __init__(self, weights: str | Path = "models/yolov8s_v2.pt", imgsz: int = 960, conf: float = 0.35,
                 device=None, ocr_model: str = DEFAULT_FASTPLATE, ocr_every: int = 1,
                 min_plate_h: int = 10, min_votes: int = 2, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.read = fastplate_reader(ocr_model)
        self.imgsz, self.conf, self.device, self.tracker = imgsz, conf, device, tracker
        self.ocr_every, self.min_plate_h, self.min_votes = max(1, ocr_every), min_plate_h, min_votes
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, min_votes: int | None = None):
        """Start a new video / image: clears tracks, votes and counts."""
        self.voter = PlateVoter(self.min_votes if min_votes is None else min_votes)
        self.frame_idx = -1
        self.vehicle_tracks: dict[str, set] = defaultdict(set)
        self.plate_vehicle: dict[int, tuple] = {}
        self._new_sequence = True

    # --------------------------------------------------------------- per frame
    def _detect(self, frame, track: bool):
        kw = dict(imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        if track:
            r = self.model.track(frame, persist=not self._new_sequence, tracker=self.tracker, **kw)[0]
        else:
            r = self.model.predict(frame, **kw)[0]
        self._new_sequence = False
        dets = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        ids = r.boxes.id.int().tolist() if (track and r.boxes.id is not None) else [None] * len(r.boxes)
        for xyxy, c, p, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids):
            dets.append({"cls": self.names[c], "conf": p, "box": [int(v) for v in xyxy], "id": tid})
        return dets

    def process(self, frame: np.ndarray, track: bool = True, draw: bool = True) -> np.ndarray:
        """Run one BGR frame. With track=False (still images) every plate gets a
        fresh id and a single read; call reset(min_votes=1) before images."""
        self.frame_idx += 1
        dets = self._detect(frame, track)
        vehicles = []
        for k, d in enumerate(dets):
            if d["id"] is None:                       # no tracking: unique id per detection
                d["id"] = 100000 * (self.frame_idx + 1) + k
            if d["cls"] != PLATE_CLASS:
                self.vehicle_tracks[d["cls"]].add(d["id"])
                vehicles.append((d["id"], d["cls"], d["box"]))

        for d in dets:
            if d["cls"] != PLATE_CLASS:
                continue
            x1, y1, x2, y2 = d["box"]
            if d["id"] not in self.plate_vehicle:
                self.plate_vehicle[d["id"]] = assign_plate_to_vehicle(d["box"], vehicles)
            if (y2 - y1) < self.min_plate_h or self.frame_idx % self.ocr_every:
                continue
            raw = self.read(crop_box(frame, d["box"], pad=2))
            self.voter.add(d["id"], correct(raw).text, weight=y2 - y1, frame=self.frame_idx)

        return self.draw(frame.copy(), dets) if draw else frame

    # ------------------------------------------------------------------ output
    def draw(self, img, dets):
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            col = COLORS.get(d["cls"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f'{d["cls"]} #{d["id"]}' if d["id"] is not None and d["id"] < 100000 else d["cls"]
            if d["cls"] == PLATE_CLASS:
                text = self.voter.result(d["id"]).text
                label = text or "plate ..."
            cv2.putText(img, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        return img

    def plates(self, include_unread: bool = False) -> list[dict]:
        out = []
        for r in self.voter.results():
            if not r.text and not include_unread:
                continue
            vid, vcls = self.plate_vehicle.get(r.track_id, (None, None))
            out.append({"plate": r.text, "agreeing_reads": r.agree, "valid_reads": r.valid_reads,
                        "ocr_reads": r.reads, "vote_share": round(r.share, 2),
                        "first_frame": r.first_frame, "last_frame": r.last_frame,
                        "plate_track": r.track_id, "vehicle_track": vid, "vehicle": vcls})
        return out

    def vehicle_counts(self) -> dict[str, int]:
        return {cls: len(ids) for cls, ids in sorted(self.vehicle_tracks.items())}
