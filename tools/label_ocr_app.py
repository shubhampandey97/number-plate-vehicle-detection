"""Small Streamlit app to hand-label plate crops for the OCR evaluation.

Shows one crop at a time (enlarged) from eval/ocr_gt/crops and writes the
answers straight into eval/ocr_gt/ocr_gt.csv (made by make_ocr_labelset.py).
Safer than Excel, which can silently turn plate text into numbers or dates.

Run from the repo root:
    streamlit run tools/label_ocr_app.py
"""
from __future__ import annotations

import base64
import csv
import io
import re
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anpr.labels import parse_line  # noqa: E402

GT = Path("eval/ocr_gt/ocr_gt.csv")
CROPS = GT.parent / "crops"
FIELDS = ["gt_text", "legible", "lines", "condition", "country"]
DATA = Path("data/final_v2")


def show(img: Image.Image, width: int, caption: str = ""):
    """Embed the image as base64 in the page (doesn't depend on Streamlit's
    media server, which some browsers/extensions block). Small crops are
    scaled with hard pixel edges so nothing is invented."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:{width}px;image-rendering:pixelated;'
        f'border:1px solid #555"/><div style="font-size:0.8em;color:#999">{caption}</div>',
        unsafe_allow_html=True)


def context_view(row):
    """The plate inside the full photo, with a red box, cropped to ~4x the plate size."""
    m = re.match(r"(.+)_p(\d+)$", Path(row["crop"]).stem)
    for split in ("test", "val", "train"):
        img_p = DATA / "images" / split / row["source_image"]
        lbl_p = DATA / "labels" / split / f"{Path(row['source_image']).stem}.txt"
        if img_p.exists() and lbl_p.exists() and m:
            lines = lbl_p.read_text().splitlines()
            k = int(m.group(2))
            box = parse_line(lines[k]) if k < len(lines) else None
            if not box:
                return None
            im = Image.open(img_p).convert("RGB")
            W, H = im.size
            _, cx, cy, w, h = box
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            ImageDraw.Draw(im).rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(255, 0, 0), width=2)
            pw, ph = max(4 * (x2 - x1), 160), max(6 * (y2 - y1), 90)
            return im.crop((int(max(0, cx * W - pw / 2)), int(max(0, cy * H - ph / 2)),
                            int(min(W, cx * W + pw / 2)), int(min(H, cy * H + ph / 2))))
    return None


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
show(img, width=min(600, max(240, img.width * 8)), caption="plate crop (enlarged, pixels not smoothed)")
ctx = context_view(r)
if ctx is not None:
    show(ctx, width=600, caption="in context (red box)")
if int(r["box_h"]) < 12:
    st.info("Very small plate (under 12 px tall). If you can't read every character, choose 'n'.")

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
