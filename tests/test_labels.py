import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.labels import parse_line  # noqa: E402


def test_box_line():
    assert parse_line("3 0.5 0.5 0.2 0.1") == (3, 0.5, 0.5, 0.2, 0.1)


def test_polygon_line_becomes_bounding_box():
    cls, cx, cy, w, h = parse_line("3 0.4 0.4 0.6 0.4 0.6 0.5 0.4 0.5")
    assert cls == 3
    assert (cx, cy, w, h) == pytest.approx((0.5, 0.45, 0.2, 0.1))


@pytest.mark.parametrize("line", ["", "3 0.5 0.5", "3 0.1 0.2 0.3 0.4 0.5"])
def test_bad_lines(line):
    assert parse_line(line) is None
