"""Pipeline logic with a fake detector and a fake plate reader (no models needed)."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _T(list):
    """Minimal stand-in for a torch tensor: .tolist() and .int()."""
    def tolist(self):
        return list(self)

    def int(self):
        return _T(int(v) for v in self)


class _Boxes:
    def __init__(self, dets, track):
        self.xyxy = _T(d[2] for d in dets)
        self.cls = _T(d[0] for d in dets)
        self.conf = _T(0.9 for _ in dets)
        self.id = _T(d[1] for d in dets) if track else None

    def __len__(self):
        return len(self.cls)


class FakeTracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeYOLO:
    """Frame k: a car (track 1) with its plate (track 2), plus a bus (track 3) from frame 2 on."""
    names = {0: "Bike", 1: "Bus", 2: "Car", 3: "Number_plate", 4: "Person", 5: "Truck", 6: "Auto"}

    def __init__(self, *_):
        self.persist_calls = []
        self.predictor = None

    def _frame(self, frame, track):
        k = int(frame[0, 0, 0])
        dets = [(2, 1, [100, 100, 300, 250]), (3, 2, [170, 200, 230, 216])]
        if k >= 2:
            dets.append((1, 3, [400, 50, 600, 300]))
        return [types.SimpleNamespace(boxes=_Boxes(dets, track))]

    def track(self, frame, **kw):
        self.persist_calls.append(kw.get("persist"))
        if self.predictor is None:                      # like Ultralytics: tracker made on first call
            self.predictor = types.SimpleNamespace(trackers=[FakeTracker()])
        return self._frame(frame, True)

    def predict(self, frame, **kw):
        return self._frame(frame, False)


def make_pipeline(monkeypatch, reads):
    fake_ul = types.ModuleType("ultralytics")
    fake_ul.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ul)
    import anpr.pipeline as pl
    it = iter(reads)
    monkeypatch.setattr(pl, "fastplate_reader", lambda *_: (lambda crop: next(it)))
    return pl.ANPRPipeline("fake.pt")


def frame(k):
    f = np.zeros((480, 640, 3), np.uint8)
    f[0, 0, 0] = k
    return f


def test_video_counts_each_vehicle_once_and_votes_plate(monkeypatch):
    # 6 frames of the same plate: 3 correct reads, 1 truncated (rejected by correction), 2 noise
    pipe = make_pipeline(monkeypatch, ["DL9CU4856", "DL9CU485", "DL9CU4856", "XQ12", "DL9CU4856", "DL9CU4855"])
    pipe.reset()
    for k in range(6):
        out = pipe.process(frame(k))
        assert out.shape == (480, 640, 3)
    assert pipe.vehicle_counts() == {"Bus": 1, "Car": 1}          # not 6 cars + 4 buses
    tracks = pipe.plates_by_track()
    assert len(tracks) == 1
    p = tracks[0]
    assert p["plate"] == "DL9CU4856" and p["agreeing_reads"] == 3 and p["valid_reads"] == 4
    assert p["ocr_reads"] == 6 and p["vehicle"] == "Car" and p["vehicle_track"] == 1
    log = pipe.plates()                      # track mode: the single DL9CU4855 read loses inside the track
    assert [(r["plate"], r["reads"], r["vehicle"], r["merged_variants"]) for r in log] == \
        [("DL9CU4856", 3, "Car", "")]
    pooled = pipe.plates(by="reads")         # reads mode: it is merged in as a 1-edit variant
    assert [(r["plate"], r["reads"], r["merged_variants"]) for r in pooled] == [("DL9CU4856", 4, "DL9CU4855")]


def test_still_image_single_read(monkeypatch):
    pipe = make_pipeline(monkeypatch, ["HR26DG7793"])
    pipe.reset(min_votes=1)
    pipe.process(frame(0), track=False)
    assert [p["plate"] for p in pipe.plates()] == ["HR26DG7793"]


def test_small_plates_are_not_read(monkeypatch):
    pipe = make_pipeline(monkeypatch, [])                       # reader must never be called
    pipe.min_plate_h = 20                                       # fake plate is 16 px tall
    pipe.reset()
    pipe.process(frame(0))
    assert pipe.plates_by_track(include_unread=True) == []


def test_class_flicker_counts_vehicle_once(monkeypatch):
    pipe = make_pipeline(monkeypatch, [""] * 10)
    pipe.reset()
    # the same track id seen as Car 3 times and Truck 2 times -> one Car
    for cls in ["Car", "Truck", "Car", "Truck", "Car"]:
        pipe.track_classes[42][cls] += 1
    pipe.track_classes[43]["Truck"] += 1                       # seen once only
    assert pipe.vehicle_counts() == {"Car": 1, "Truck": 1}
    assert pipe.vehicle_counts(min_frames=2) == {"Car": 1}


def test_tracker_persists_and_is_reset_between_videos(monkeypatch):
    # Ultralytics fixes `persist` on the first track() call; the pipeline must always
    # pass persist=True and reset the tracker itself between videos.
    pipe = make_pipeline(monkeypatch, [""] * 20)
    pipe.reset()
    for k in range(3):
        pipe.process(frame(k))
    assert pipe.model.persist_calls == [True, True, True]
    tracker = pipe.model.predictor.trackers[0]
    assert tracker.resets == 0
    pipe.reset()                                         # next video
    assert tracker.resets == 1
