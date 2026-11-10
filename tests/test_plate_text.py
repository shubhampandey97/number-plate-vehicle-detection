import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.plate_text import cer, clean, correct, is_valid, join_lines  # noqa: E402


@pytest.mark.parametrize("plate", ["MH12AB1234", "DL3CAF0001", "UP65T1234", "KA011234", "22BH1234AA"])
def test_valid_plates(plate):
    assert is_valid(plate)


@pytest.mark.parametrize("plate", ["NAI3MRU", "DS77", "XX12AB1234", "MH00AB1234", "MH12AB0000", ""])
def test_invalid_plates(plate):
    assert not is_valid(plate)


def test_clean():
    assert clean("mh 12-ab.1234") == "MH12AB1234"


@pytest.mark.parametrize("raw,expected", [
    ("MH12AB1234", "MH12AB1234"),    # already valid
    ("MHI2AB1234", "MH12AB1234"),    # I -> 1 in RTO position
    ("MH12A81234", "MH12AB1234"),    # 8 -> B in series position
    ("DL3CAF000I", "DL3CAF0001"),    # I -> 1 in number
])
def test_correct(raw, expected):
    c = correct(raw)
    assert c.valid and c.text == expected


def test_single_digit_rto_only_for_delhi():
    assert is_valid("DL3CAF0001")
    assert not is_valid("HR2GEA7905")            # must not be read as HR + 2 + GEA
    c = correct("HR2GEA7905")                     # real EasyOCR output in eval/ocr_gt
    assert c.valid and c.text == "HR26EA7905"


@pytest.mark.parametrize("raw", ["DL9CU485", "DL1TT055", "AP16D043"])
def test_correct_does_not_invent_plates_from_truncated_reads(raw):
    # real fast-plate-ocr outputs (true plates DL9CU4856, DL1RTC0583, AP16TD4436);
    # the old rules "repaired" them into DL9C0485, DL1T7055, AP160043
    assert not correct(raw).valid


def test_plate_without_series_only_accepted_as_read():
    assert correct("HR743571").text == "HR743571"     # exact read, kept
    assert not correct("HR74357I").valid              # would need a substitution, rejected


def test_correct_rejects_unknown_state():
    # 0 -> O gives 'OL', which is not a state code: reject rather than guess
    assert not correct("0L3CAF0001").valid


def test_correct_refuses_too_many_changes():
    c = correct("NAI3MRU")          # real line from outputs/logs/plate_logs.csv
    assert not c.valid and c.text == ""


def test_join_two_line_plate():
    # bike plate: top line 'MH12', bottom line 'AB1234'; EasyOCR returns both
    frags = [
        ([[0, 40], [90, 40], [90, 70], [0, 70]], "AB1234", 0.9),
        ([[10, 0], [70, 0], [70, 30], [10, 30]], "MH12", 0.6),
    ]
    text, conf = join_lines(frags)
    assert text == "MH12AB1234"
    assert 0.6 < conf < 0.9


def test_join_single_line_left_to_right():
    frags = [([[60, 0], [120, 0], [120, 20], [60, 20]], "AB1234", 0.8),
             ([[0, 2], [50, 2], [50, 22], [0, 22]], "MH12", 0.8)]
    assert join_lines(frags)[0] == "MH12AB1234"


def test_cer():
    assert cer("MH12AB1234", "MH12AB1234") == 0
    assert cer("MH12AB123", "MH12AB1234") == pytest.approx(0.1)
    assert cer("", "AB") == 1.0


@pytest.mark.parametrize("plate", ["GX15OGJ", "NA13NRU", "AP05JEO", "FJ14ZHY", "CA02FKD"])
def test_uk_plates_valid(plate):
    # real plates visible in data/Traffic Dataset/video11.mp4 (British footage)
    assert is_valid(plate, "UK") and not is_valid(plate, "IN")


def test_uk_correction_fixes_digit_letter_confusions():
    assert correct("GXI5OGJ", region="UK").text == "GX15OGJ"     # I -> 1 in the age digits
    assert correct("NAI3NRU", region="UK").text == "NA13NRU"
    assert not correct("GXIS0GJ", region="UK").valid              # old EasyOCR read: needs 3 fixes > max 2
    assert not correct("NAI3MRUX", region="UK").valid             # wrong length: no guessing


def test_any_region_has_no_format_rules():
    assert correct("4345CBK", region="ANY").text == "4345CBK"     # Mexican plate
    assert not correct("DS77", region="ANY").valid                # too short
    with pytest.raises(ValueError):
        correct("MH12AB1234", region="XX")
