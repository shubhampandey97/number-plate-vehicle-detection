import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.voting import PlateVoter, assign_plate_to_vehicle, edit_distance  # noqa: E402


def test_correct_read_wins_over_scattered_errors():
    v = PlateVoter(min_votes=2)
    # 8 frames of one Delhi car: 3 correct reads, 2 different wrong-but-valid, 3 rejected
    for f, (txt, h) in enumerate([("DL9CU4856", 14), ("", 12), ("DL9CU4855", 13), ("DL9CU4856", 16),
                                  ("", 11), ("DL9CU4858", 12), ("DL9CU4856", 18), ("", 10)]):
        v.add(7, txt, h, f)
    r = v.result(7)
    assert r.text == "DL9CU4856" and r.agree == 3 and r.valid_reads == 5 and r.reads == 8
    assert (r.first_frame, r.last_frame) == (0, 7)


def test_single_read_not_enough_on_video():
    v = PlateVoter(min_votes=2)
    v.add(1, "HR26DG7793", 40, 0)
    assert v.result(1).text == ""          # one frame only: not reported
    v1 = PlateVoter(min_votes=1)           # still image mode
    v1.add(1, "HR26DG7793", 40, 0)
    assert v1.result(1).text == "HR26DG7793"


def test_weight_breaks_ties_by_plate_size():
    v = PlateVoter(min_votes=1)
    v.add(3, "HR26EA7905", 30, 0)          # big, clear
    v.add(3, "HR26EA7906", 10, 1)          # small, far away
    assert v.result(3).text == "HR26EA7905"


def test_no_valid_reads():
    v = PlateVoter()
    v.add(5, "", 20, 0)
    r = v.result(5)
    assert r.text == "" and r.valid_reads == 0 and r.reads == 1


def test_assign_plate_to_smallest_containing_vehicle():
    vehicles = [(1, "Bus", (0, 0, 400, 300)), (2, "Car", (100, 100, 200, 200)), (3, "Bike", (500, 0, 600, 100))]
    assert assign_plate_to_vehicle((140, 170, 160, 180), vehicles) == (2, "Car")
    assert assign_plate_to_vehicle((700, 700, 720, 710), vehicles) == (None, None)


def test_plate_log_survives_id_swaps():
    # video11 situation: track 1 starts on GX15OGJ, then jumps to the AP05JEO car;
    # track 2 is AP05JEO from the start; NA13NFU is a 1-edit misread of NA13NRU
    v = PlateVoter(min_votes=2)
    for f in range(10):
        v.add(1, "GX15OGJ" if f < 5 else "AP05JEO", 20, f)
        v.add(2, "AP05JEO", 20, f)
    for f in range(6):
        v.add(3, "NA13NRU", 30, f)
    v.add(4, "NA13NFU", 10, 3)
    v.add(4, "NA13NFU", 10, 4)
    log = {r["plate"]: r for r in v.plate_log(min_reads=3)}
    assert set(log) == {"GX15OGJ", "AP05JEO", "NA13NRU"}       # each plate once, no NA13NFU
    assert log["AP05JEO"]["reads"] == 15 and log["AP05JEO"]["tracks"] == [1, 2]
    assert log["NA13NRU"]["merged_variants"] == ["NA13NFU"] and log["NA13NRU"]["reads"] == 8
    assert (log["GX15OGJ"]["first_frame"], log["GX15OGJ"]["last_frame"]) == (0, 4)


def test_plate_log_min_reads():
    v = PlateVoter()
    v.add(1, "HR26DG7793", 40, 0)
    v.add(1, "HR26DG7793", 40, 1)
    assert v.plate_log(min_reads=3) == []
    assert [r["plate"] for r in v.plate_log(min_reads=2)] == ["HR26DG7793"]


def test_edit_distance():
    assert edit_distance("NA13NRU", "NA13NFU") == 1
    assert edit_distance("GX15OGJ", "AP05JEO") > 1
