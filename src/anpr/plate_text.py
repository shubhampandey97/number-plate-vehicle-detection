"""Plate-text post-processing for Indian number plates.

Pure Python (no OpenCV / EasyOCR), so it is fast to unit-test.

Formats handled
---------------
* Standard:  SS DD [L{0,3}] NNNN   e.g. MH12AB1234, DL3CAF0001, UP65T1234
  SS = state/UT code, DD = RTO code (1-2 digits), L = series letters, N = number.
* Bharat (BH) series:  YY BH NNNN L{1,2}   e.g. 22BH1234AA
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product

ALNUM = re.compile(r"[^A-Z0-9]")

STATE_CODES = frozenset(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TN TR TS UK UA UP WB CT".split()
)

STANDARD_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
BH_RE = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")

# Common OCR confusions: what a character is likely to be when the position
# requires the other type.
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "T": "7", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


def clean(text: str) -> str:
    """Upper-case and keep only A-Z / 0-9."""
    return ALNUM.sub("", text.upper())


def is_valid(plate: str) -> bool:
    """True if `plate` (already cleaned) matches an Indian plate format."""
    m = STANDARD_RE.match(plate)
    if m:
        return m.group(1) in STATE_CODES and m.group(2) != "00" and m.group(4) != "0000"
    return bool(BH_RE.match(plate))


@dataclass(frozen=True)
class Correction:
    text: str          # corrected text ('' if no valid reading found)
    raw: str           # cleaned OCR text
    n_subs: int        # number of character substitutions applied
    valid: bool        # does `text` match a plate format?


def _templates(n: int):
    """Yield type-templates ('L'/'D' per char) of length n for valid formats."""
    for rto, series in product((1, 2), (0, 1, 2, 3)):
        if 2 + rto + series + 4 == n:
            yield "LL" + "D" * rto + "L" * series + "DDDD"
    if n in (9, 10):  # BH: YY BH NNNN L{1,2}
        yield "DDLL" + "DDDD" + "L" * (n - 8)


def _coerce(text: str, template: str):
    out, subs = [], 0
    for ch, kind in zip(text, template):
        if kind == "L" and ch.isdigit():
            if ch not in TO_LETTER:
                return None
            ch, subs = TO_LETTER[ch], subs + 1
        elif kind == "D" and ch.isalpha():
            if ch not in TO_DIGIT:
                return None
            ch, subs = TO_DIGIT[ch], subs + 1
        out.append(ch)
    return "".join(out), subs


def correct(text: str, max_subs: int = 2) -> Correction:
    """Fix position-type OCR confusions (O<->0, I<->1, S<->5, B<->8 ...).

    Tries every valid template of the same length and keeps the valid reading
    needing the fewest substitutions. Never changes the length, never invents
    characters, and gives up (valid=False) if more than `max_subs` changes
    would be needed - better an honest miss than a confident wrong plate.
    """
    raw = clean(text)
    if is_valid(raw):
        return Correction(raw, raw, 0, True)
    best = None
    for tpl in _templates(len(raw)):
        res = _coerce(raw, tpl)
        if res and res[1] <= max_subs and is_valid(res[0]):
            if best is None or res[1] < best[1]:
                best = res
    if best:
        return Correction(best[0], raw, best[1], True)
    return Correction("", raw, 0, False)


def join_lines(fragments: list[tuple[list, str, float]]) -> tuple[str, float]:
    """Join EasyOCR fragments of a (possibly two-line) plate in reading order.

    `fragments` are EasyOCR `readtext(detail=1)` results: (box, text, conf).
    Two-wheeler and auto plates in India are usually two lines; picking only
    the single most confident fragment (the old behaviour) drops half the plate.
    Rows are grouped by vertical overlap, then read top-to-bottom, left-to-right.
    Returns (text, mean confidence weighted by fragment length).
    """
    items = []
    for box, txt, conf in fragments:
        t = clean(txt)
        if not t:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), max(ys), min(xs), t, float(conf)))
    if not items:
        return "", 0.0
    items.sort(key=lambda r: (r[0] + r[1]) / 2)
    rows: list[list] = []
    for it in items:
        cy = (it[0] + it[1]) / 2
        if rows and rows[-1][0][0] <= cy <= rows[-1][0][1]:
            rows[-1].append(it)
        else:
            rows.append([it])
    ordered = [it for row in rows for it in sorted(row, key=lambda r: r[2])]
    text = "".join(it[3] for it in ordered)
    total = sum(len(it[3]) for it in ordered)
    conf = sum(len(it[3]) * it[4] for it in ordered) / total
    return text, conf


def cer(pred: str, gt: str) -> float:
    """Character error rate = Levenshtein(pred, gt) / len(gt)."""
    if not gt:
        return float(bool(pred))
    prev = list(range(len(gt) + 1))
    for i, a in enumerate(pred, 1):
        cur = [i]
        for j, b in enumerate(gt, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(gt)
