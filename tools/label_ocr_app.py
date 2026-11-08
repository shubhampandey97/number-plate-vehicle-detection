"""Small Streamlit app to hand-label plate crops for the OCR evaluation.

Shows one crop at a time (enlarged) from eval/ocr_gt/crops and writes the
answers straight into eval/ocr_gt/ocr_gt.csv (made by make_ocr_labelset.py).
Safer than Excel, which can silently turn plate text into numbers or dates.

Run from the repo root:
    streamlit run tools/label_ocr_app.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import streamlit as st
from PIL import Image

GT = Path("eval/ocr_gt/ocr_gt.csv")
CROPS = GT.parent / "crops"
FIELDS = ["gt_text", "legible", "lines", "condition", "country"]


def load():
    with open(GT, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            r.setdefault(k, "")
    return rows


def save(rows):
    tmp = GT.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(GT)


def is_done(r):
    return r["legible"].strip().lower() == "n" or bool(r["gt_text"].strip())


st.set_page_config(page_title="OCR labelling", layout="centered")
if not GT.exists():
    st.error(f"{GT} not found. Run: python tools/make_ocr_labelset.py")
    st.stop()

rows = load()
done = sum(is_done(r) for r in rows)
st.progress(done / len(rows), text=f"{done} / {len(rows)} labelled")

if "i" not in st.session_state:
    st.session_state.i = next((k for k, r in enumerate(rows) if not is_done(r)), 0)
i = st.session_state.i
r = rows[i]

st.caption(f"#{i + 1}  {r['crop']}  (from {r['source_image']}, {r['box_w']}×{r['box_h']} px)")
img = Image.open(CROPS / r["crop"])
scale = max(1, int(300 / max(img.width, 1)))
st.image(img.resize((img.width * scale, img.height * scale), Image.NEAREST))

with st.form(f"f{i}"):
    text = st.text_input("Plate text (letters and digits only, no spaces; leave empty if unreadable)",
                         value=r["gt_text"])
    c1, c2, c3 = st.columns(3)
    legible = c1.radio("Readable by a human?", ["y", "n"], index=1 if r["legible"] == "n" else 0)
    lines = c2.radio("Lines", ["1", "2"], index=1 if r["lines"] == "2" else 0)
    country = c3.radio("Country", ["IN", "MX", "other"],
                       index=["IN", "MX", "other"].index(r["country"]) if r["country"] in ("IN", "MX", "other") else 0)
    cond = st.multiselect("Conditions (optional)", ["day", "night", "blur", "angle", "small", "dirty", "partial"],
                          default=[c for c in r["condition"].split("+") if c])
    go = st.form_submit_button("Save & next ▶")

if go:
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if legible == "y" and not clean:
        st.warning("Readable plates need the text. Choose 'n' if you can't read it.")
    else:
        r.update(gt_text=clean if legible == "y" else "", legible=legible, lines=lines,
                 country=country, condition="+".join(cond))
        save(rows)
        st.session_state.i = min(i + 1, len(rows) - 1)
        st.rerun()

b1, b2, b3 = st.columns(3)
if b1.button("◀ Back") and i > 0:
    st.session_state.i = i - 1
    st.rerun()
if b2.button("Skip ▶") and i < len(rows) - 1:
    st.session_state.i = i + 1
    st.rerun()
if b3.button("Next unlabelled"):
    st.session_state.i = next((k for k, x in enumerate(rows) if not is_done(x)), i)
    st.rerun()
