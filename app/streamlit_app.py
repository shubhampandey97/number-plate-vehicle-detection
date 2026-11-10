"""AI Traffic Monitoring & ANPR dashboard.

Runs the shared pipeline in src/anpr/pipeline.py, the same code that
tools/run_video.py and the tests use:
YOLOv8s (split v2, 960 px) + ByteTrack + plate-trained OCR + Indian-format
check + multi-frame voting. Vehicles are counted once per track.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anpr.pipeline import ANPRPipeline  # noqa: E402

MODELS = {
    "YOLOv8s, split v2 (recommended)": "models/yolov8s_v2.pt",
    "YOLOv8l, split v2": "models/yolov8l_v2.pt",
    "YOLOv8l, original (leaked split)": "models/yolov8l_baseline.pt",
}

st.set_page_config(page_title="AI Traffic Monitoring & ANPR", page_icon="🚗", layout="wide")
st.title("🚗 AI Traffic Monitoring & ANPR")
st.caption("Held-out test: detection mAP50 0.743 (YOLOv8s). Plate reading on hand-labelled test plates: "
           "67 % of logged plates correct after format check; plates under ~20 px are usually unreadable. "
           "See docs/results/ in the repo.")


@st.cache_resource(show_spinner="Loading models ...")
def load_pipeline(weights: str, imgsz: int) -> ANPRPipeline:
    return ANPRPipeline(ROOT / weights, imgsz=imgsz)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Detector", [m for m in MODELS if (ROOT / MODELS[m]).exists()] or list(MODELS))
imgsz = st.sidebar.select_slider("Image size", [640, 960, 1280], value=960,
                                 help="The models were trained at 960 px. Smaller is faster but misses small plates.")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
min_votes = st.sidebar.slider("Agreeing reads needed per plate (video)", 1, 5, 2,
                              help="A plate is logged only when this many frames give the same valid reading.")
ocr_every = st.sidebar.slider("Read plates every N frames", 1, 5, 1, help="Higher is faster, fewer votes.")
show_every = st.sidebar.slider("Show every N-th frame (video)", 1, 10, 3, help="Display only; all frames are processed.")
region = st.sidebar.selectbox("Plate format", ["IN", "UK", "ANY"],
                              help="Format rules for the plate check. IN = Indian, UK = British "
                                   "(the 'Traffic Dataset' videos), ANY = no format rules.")
mode = st.sidebar.radio("Mode", ["Image", "Video", "Live camera"])

if not (ROOT / MODELS[model_name]).exists():
    st.error(f"Model file not found: {MODELS[model_name]}. See models/README.md.")
    st.stop()
pipe = load_pipeline(MODELS[model_name], imgsz)
pipe.conf, pipe.ocr_every, pipe.region = conf, ocr_every, region


def show_results():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Vehicles (counted once per track)")
        counts = pipe.vehicle_counts(min_frames=1 if mode == "Image" else 5)
        st.dataframe(pd.DataFrame(counts.items(), columns=["Vehicle", "Count"]), use_container_width=True)
    with c2:
        st.subheader("Plates")
        plates = pd.DataFrame(pipe.plates())
        if plates.empty:
            st.info("No plate reached enough agreeing, format-valid reads. Check the plate format "
                    "(IN/UK/ANY) in the sidebar; very small or blurred plates can't be read.")
        else:
            st.dataframe(plates, use_container_width=True)
            st.download_button("Download plates CSV", plates.to_csv(index=False).encode(), "plates.csv", "text/csv")
        with st.expander("Per plate track (diagnostic; tracker ids can swap between cars)"):
            st.dataframe(pd.DataFrame(pipe.plates_by_track(include_unread=True)), use_container_width=True)


# ------------------------------------------------------------------- modes
if mode == "Image":
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        pipe.reset(min_votes=1)
        out = pipe.process(img, track=False)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        show_results()

elif mode == "Video":
    up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if up is not None and st.button("Process video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as f:
            f.write(up.read())
        cap = cv2.VideoCapture(f.name)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        pipe.reset(min_votes=min_votes)
        view, bar, stat = st.empty(), st.progress(0.0), st.empty()
        k, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw = k % show_every == 0
            out = pipe.process(frame, track=True, draw=draw)
            if draw:
                view.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
            k += 1
            bar.progress(min(1.0, k / n))
            stat.caption(f"frame {k}/{n} · {k / (time.time() - t0):.1f} fps · plates read: {len(pipe.plates())}")
        cap.release()
        show_results()

else:
    st.subheader("Live camera")
    run = st.checkbox("Start camera")
    view = st.empty()
    if run:
        pipe.reset(min_votes=min_votes)
        cap = cv2.VideoCapture(0)
        while run:
            ok, frame = cap.read()
            if not ok:
                st.error("Unable to access the camera")
                break
            view.image(cv2.cvtColor(pipe.process(frame), cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()
    show_results()
