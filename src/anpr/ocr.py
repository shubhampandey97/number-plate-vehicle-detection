"""Single OCR implementation for the whole repo.

Before this, the same crop -> resize -> bilateral -> Otsu -> EasyOCR code was
copy-pasted in plate_ocr.py, video_ocr.py, vehicle_tracking.py and the
Streamlit app, each with slightly different settings. Everything should call
`read_plate()` so that the evaluated pipeline is the deployed pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .plate_text import Correction, correct, is_valid, join_lines

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VARIANTS = ("gray", "otsu", "adaptive", "clahe")


def crop_box(image: np.ndarray, xyxy, pad: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    return image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def preprocess(crop: np.ndarray, variant: str = "otsu", target_h: int = 96) -> np.ndarray:
    """Resize to a fixed height (not a fixed 4x: a 15-px plate and a 120-px
    plate need very different scaling), then apply the chosen variant.
    'otsu' reproduces the original pipeline."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = target_h / max(gray.shape[0], 1)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    if variant == "gray":
        return gray
    if variant == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(gray)
    if variant == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


@dataclass(frozen=True)
class PlateRead:
    text: str          # final text ('' if rejected)
    raw: str           # joined OCR text before correction
    conf: float        # OCR confidence (length-weighted mean)
    valid: bool        # matches an Indian plate format
    n_subs: int        # position-aware substitutions applied


def read_plate(reader, crop: np.ndarray, variant: str = "otsu",
               use_correction: bool = True, min_conf: float = 0.0) -> PlateRead:
    """Run EasyOCR on one plate crop. `reader` is an easyocr.Reader (passed in
    so tests and the app control when the heavy model is loaded)."""
    if crop is None or crop.size == 0:
        return PlateRead("", "", 0.0, False, 0)
    img = preprocess(crop, variant)
    frags = reader.readtext(img, allowlist=ALLOWLIST, paragraph=False, detail=1)
    frags = [f for f in frags if f[2] >= min_conf]
    raw, conf = join_lines(frags)
    if use_correction:
        c: Correction = correct(raw)
        return PlateRead(c.text, raw, conf, c.valid, c.n_subs)
    return PlateRead(raw, raw, conf, is_valid(raw), 0)
