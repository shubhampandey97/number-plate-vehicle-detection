"""Plate-text post-processing: validation and OCR-confusion correction.

Regions (`region=` argument): "IN" (default, Indian formats), "UK" (current
British format), "ANY" (no format rules: just clean the text). The data mixes
countries: the "Traffic Dataset" videos are British, some images are Mexican.

Pure Python (no OpenCV / EasyOCR), so it is fast to unit-test.

Formats handled
---------------
* Standard:  SS DD [L{0,3}] NNNN   e.g. MH12AB1234, DL3CAF0001, UP65T1234
  SS = state/UT code, DD = RTO code, L = series letters, N = number.
  RTO codes are written with 2 digits (MH01, HR26) everywhere except Delhi,
  which uses 1-2 digits followed by a category letter (DL3C..., DL10C...).
  Without that rule "HR2GEA7905" was accepted as HR + 2 + GEA + 7905 instead
  of being corrected to HR26EA7905.
  Delhi plates carry a category letter plus series (DL9CU4856, DL1RTC0583),
  so at least 2 letters follow the RTO digits.

`correct()` is stricter than `is_valid()`: it never *creates* a rare format
by substitution. A truncated read like "DL9CU485" must not be "repaired" into
the valid-looking but wrong "DL9C0485" (seen in the eval). Plates without
series letters (old "HR743571") are accepted only exactly as read.
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
# UK since 2001: area (2 letters) + age (2 digits) + 3 random letters, e.g. GX15OGJ.
# I, Q and Z are not used in the first two letters; I and Q not in the last three.
UK_RE = re.compile(r"^[A-HJ-PR-Y]{2}\d{2}[A-HJ-PR-Z]{3}$")
REGIONS = ("IN", "UK", "ANY")

# Common OCR confusions: what a character is likely to be when the position
# requires the other type.
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "T": "7", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


def clean(text: str) -> str:
    """Upper-case and keep only A-Z / 0-9."""
    return ALNUM.sub("", text.upper())


def is_valid(plate: str, region: str = "IN") -> bool:
    """True if `plate` (already cleaned) matches a plate format of `region`."""
    if region == "UK":
        return bool(UK_RE.match(plate))
    if region == "ANY":
        return 5 <= len(plate) <= 10 and any(c.isdigit() for c in plate) and any(c.isalpha() for c in plate)
    m = STANDARD_RE.match(plate)
    if m:
        state, rto, series, num = m.groups()
        if state not in STATE_CODES or rto in ("0", "00") or num == "0000":
            return False
        if state == "DL":
            return len(series) >= 2
        return len(rto) == 2
    return bool(BH_RE.match(plate))


@dataclass(frozen=True)
class Correction:
    text: str          # corrected text ('' if no valid reading found)
    raw: str           # cleaned OCR text
    n_subs: int        # number of character substitutions applied
    valid: bool        # does `text` match a plate format?


def _templates(n: int, region: str = "IN"):
    """Yield type-templates ('L'/'D' per char) of length n for valid formats."""
    if region == "UK":
        if n == 7:
            yield "LLDDLLL"
        return
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


def correct(text: str, max_subs: int = 2, region: str = "IN") -> Correction:
    """Fix position-type OCR confusions (O<->0, I<->1, S<->5, B<->8 ...).

    Tries every valid template of the same length and keeps the valid reading
    needing the fewest substitutions. Never changes the length, never invents
    characters, and gives up (valid=False) if more than `max_subs` changes
    would be needed - better an honest miss than a confident wrong plate.
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    raw = clean(text)
    if is_valid(raw, region):
        return Correction(raw, raw, 0, True)
    if region == "ANY":
        return Correction("", raw, 0, False)
    best = None
    for tpl in _templates(len(raw), region):
        if tpl.startswith("LL") and "L" not in tpl[2:]:
            continue                      # no series letters: only accepted exactly as read
        res = _coerce(raw, tpl)
        if res and res[1] <= max_subs and is_valid(res[0], region):
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
