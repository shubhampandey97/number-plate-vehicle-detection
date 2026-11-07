import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.ocr import VARIANTS, preprocess, read_plate  # noqa: E402


class FakeReader:
    """Stands in for easyocr.Reader: returns a two-line bike plate."""
    def readtext(self, img, **kw):
        assert img.ndim == 2 and img.shape[0] == 96
        return [([[0, 50], [90, 50], [90, 90], [0, 90]], "AB1Z34", 0.9),
                ([[5, 0], [80, 0], [80, 40], [5, 40]], "MHI2", 0.7)]


@pytest.mark.parametrize("variant", VARIANTS)
def test_preprocess_fixed_height(variant):
    crop = np.random.default_rng(0).integers(0, 255, (24, 80, 3), dtype=np.uint8)
    assert preprocess(crop, variant).shape[0] == 96


def test_read_plate_joins_and_corrects():
    crop = np.zeros((30, 60, 3), np.uint8)
    r = read_plate(FakeReader(), crop)
    assert r.raw == "MHI2AB1Z34"
    assert r.text == "MH12AB1234" and r.valid and r.n_subs == 2


def test_read_plate_empty_crop():
    assert read_plate(FakeReader(), np.zeros((0, 0, 3), np.uint8)).text == ""
