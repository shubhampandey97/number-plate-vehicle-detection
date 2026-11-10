import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.voting import PlateVoter, assign_plate_to_vehicle  # noqa: E402


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
