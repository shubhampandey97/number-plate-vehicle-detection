"""Single OCR implementation for the whole repo.

Before this, the same crop -> resize -> bilateral -> Otsu -> EasyOCR code was
copy-pasted in plate_ocr.py, video_ocr.py, vehicle_tracking.py and the
Streamlit app, each with slightly different settings. Everything should call
this module so that the evaluated pipeline is the deployed pipeline.

Two readers:
* `fastplate_reader()`: plate-trained recogniser (fast-plate-ocr), the default
  for the app and tracker since 8 Nov (6/37 vs 2/37 exact on test crops,
  67 % precision after format correction).
* `read_plate(easyocr_reader, crop)`: the EasyOCR path, kept for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .plate_text import Correction, clean, correct, is_valid, join_lines

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VARIANTS = ("gray", "otsu", "adaptive", "clahe")
DEFAULT_FASTPLATE = "cct-s-v2-global-model"


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


def _patch_fastplate_download(tries: int = 8):
    """fast-plate-ocr downloads its model from GitHub releases with a single
    urllib call, which fails on a transient 504. Retry with back-off and a
    browser-like User-Agent; if it still fails, print the URL and the target
    path so the file can be downloaded by hand and dropped in place."""
    import time
    import urllib.request

    from fast_plate_ocr.inference import hub

    def download(url, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for k in range(tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                tmp = filename.with_suffix(filename.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(filename)
                print(f"downloaded {filename.name} ({len(data) / 1e6:.1f} MB)")
                return
            except Exception as e:  # noqa: BLE001
                last = e
                wait = min(60, 5 * 2 ** k)
                print(f"download failed ({e}); retry {k + 1}/{tries} in {wait}s")
                time.sleep(wait)
        raise SystemExit(f"\nCould not download after {tries} tries ({last}).\n"
                         f"Download this URL in your browser:\n  {url}\n"
                         f"and save it exactly as:\n  {filename}\nThen run the command again.")

    hub._download_with_progress = download


def fastplate_reader(model_name: str = DEFAULT_FASTPLATE):
    """Plate-specific recogniser (fast-plate-ocr, ONNX; trained on plates, not general text).
    Returns a function crop(BGR ndarray) -> cleaned raw text (no format correction).
    On the hand-labelled test crops it beat EasyOCR clearly: CER 0.23 vs 0.87
    (docs/results/ocr_eval.md)."""
    import tempfile

    from fast_plate_ocr import LicensePlateRecognizer

    _patch_fastplate_download()
    m = LicensePlateRecognizer(model_name)
    tmp = Path(tempfile.mkdtemp()) / "crop.png"

    def read(crop):
        if crop is None or crop.size == 0:
            return ""
        cv2.imwrite(str(tmp), crop)          # pass a file: the library handles its own colour/resize
        out = m.run(str(tmp))
        p = out[0] if isinstance(out, (list, tuple)) and out else out
        return clean(str(getattr(p, "plate", p)))   # v1.1 returns objects with .plate, older versions str
    return read
