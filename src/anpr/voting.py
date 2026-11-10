"""Multi-frame plate voting and plate-to-vehicle association (pure Python).

A tracked plate is visible in many frames. A single-frame OCR read is often
wrong (zero-shot fast-plate-ocr: 16 % exact on test crops), but wrong reads
rarely agree with each other, while correct reads repeat. So:

* only format-valid reads (after `plate_text.correct`) are counted,
* each read is weighted by plate height in pixels (bigger plate, more reliable),
* a track's plate is the text with the most weight, and it is reported only if
  at least `min_votes` reads agree. A single lucky (or unlucky) frame is not
  enough on video; use min_votes=1 for still images.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TrackVotes:
    reads: int = 0                                   # OCR attempts on this track
    votes: dict = field(default_factory=lambda: defaultdict(float))   # text -> weight
    counts: dict = field(default_factory=lambda: defaultdict(int))    # text -> n reads
    first_frame: int = -1
    last_frame: int = -1


@dataclass(frozen=True)
class PlateResult:
    track_id: int
    text: str            # '' if no text reached min_votes
    agree: int           # reads that agree with `text`
    valid_reads: int     # format-valid reads on this track
    reads: int           # all OCR attempts on this track
    share: float         # weight share of the winning text among valid reads
    first_frame: int
    last_frame: int


class PlateVoter:
    def __init__(self, min_votes: int = 2):
        self.min_votes = min_votes
        self.tracks: dict[int, TrackVotes] = defaultdict(TrackVotes)

    def add(self, track_id: int, text: str, weight: float, frame: int) -> None:
        """Record one OCR attempt. `text` is the corrected text ('' if rejected)."""
        t = self.tracks[track_id]
        t.reads += 1
        t.first_frame = frame if t.first_frame < 0 else t.first_frame
        t.last_frame = frame
        if text:
            t.votes[text] += max(weight, 1e-6)
            t.counts[text] += 1

    def result(self, track_id: int) -> PlateResult:
        t = self.tracks.get(track_id)            # .get: asking must not create an entry
        if t is None:
            return PlateResult(track_id, "", 0, 0, 0, 0.0, -1, -1)
        valid = sum(t.counts.values())
        if not t.votes:
            return PlateResult(track_id, "", 0, 0, t.reads, 0.0, t.first_frame, t.last_frame)
        text = max(t.votes, key=lambda k: (t.votes[k], t.counts[k]))
        agree = t.counts[text]
        share = t.votes[text] / sum(t.votes.values())
        if agree < self.min_votes:
            text = ""
        return PlateResult(track_id, text, agree, valid, t.reads, share, t.first_frame, t.last_frame)

    def results(self) -> list[PlateResult]:
        return [self.result(tid) for tid in sorted(self.tracks)]


def contains(outer, inner_center) -> bool:
    x1, y1, x2, y2 = outer
    cx, cy = inner_center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def assign_plate_to_vehicle(plate_box, vehicles):
    """vehicles: list of (track_id, class_name, xyxy). Returns (track_id, class_name)
    of the smallest vehicle box containing the plate centre, or (None, None)."""
    cx, cy = (plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2
    best, best_area = (None, None), None
    for tid, cls, box in vehicles:
        if contains(box, (cx, cy)):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if best_area is None or area < best_area:
                best, best_area = (tid, cls), area
    return best
