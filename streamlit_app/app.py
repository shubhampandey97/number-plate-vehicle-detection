import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import pandas as pd
import tempfile
import re
from collections import defaultdict


# PAGE CONFIG
st.set_page_config(
    page_title="AI Traffic Monitoring System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI Traffic Monitoring & ANPR System")


# LOAD MODELS
@st.cache_resource
def load_models():
    model = YOLO(
        r"runs\detect\runs1\detect\traffic_object_detector-2\weights\best.pt"
    )

    reader = easyocr.Reader(
        ['en'],
        gpu=False
    )

    return model, reader


model, reader = load_models()


# CLASS NAMES
CLASS_NAMES = [
    "Bike",
    "Bus",
    "Car",
    "Number_plate",
    "Person",
    "Truck",
    "Auto"
]


# SESSION STATE
if "vehicle_counts" not in st.session_state:
    st.session_state.vehicle_counts = defaultdict(int)

if "detected_plates" not in st.session_state:
    st.session_state.detected_plates = []

if "processed_track_ids" not in st.session_state:
    st.session_state.processed_track_ids = set()


# SIDEBAR SETTINGS
st.sidebar.header("⚙ Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.5,
    0.05
)

ocr_threshold = st.sidebar.slider(
    "OCR Confidence",
    0.1,
    1.0,
    0.3,
    0.05
)

# RESET BUTTON
if st.sidebar.button("Reset Analytics"):

    st.session_state.vehicle_counts = defaultdict(int)

    st.session_state.detected_plates = []

    st.session_state.processed_track_ids = set()

    st.success("Analytics Reset Successfully")


# MODE SELECTION
mode = st.sidebar.radio(
    "Select Mode",
    [
        "Image",
        "Video",
        "Live Camera"
    ]
)

uploaded_file = None

if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

elif mode == "Video":
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4"]
    )


# PROCESS IMAGE FUNCTION
def process_image(image):
    original_image = image.copy()

    results = model(
        image,
        verbose=False
    )[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])

        conf = float(box.conf[0])

        if conf < confidence_threshold:
            continue

        class_name = CLASS_NAMES[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # VEHICLE COUNTING
        unique_id = f"{class_name}_{x1}_{y1}"

        if unique_id not in st.session_state.processed_track_ids:
            if class_name != "Number_plate":
                st.session_state.vehicle_counts[class_name] += 1

            st.session_state.processed_track_ids.add(unique_id)

        # DRAW DETECTION
        color = (0, 255, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{class_name} {conf:.2f}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # OCR FOR NUMBER PLATES
        if class_name == "Number_plate":
            padding = 5

            plate_crop = original_image[
                max(0, y1-padding):min(original_image.shape[0], y2+padding),
                max(0, x1-padding):min(original_image.shape[1], x2+padding)
            ]

            if plate_crop.size == 0:
                continue

            # PREPROCESSING
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.bilateralFilter(
                gray,
                11,
                17,
                17
            )

            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # OCR
            ocr_results = reader.readtext(
                thresh,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                detail=1
            )

            best_text = ""
            best_score = 0

            for result in ocr_results:
                text = result[1]

                score = result[2]

                if score < ocr_threshold:
                    continue

                cleaned_text = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )

                if len(cleaned_text) < 4:
                    continue

                if score > best_score:
                    best_score = score

                    best_text = cleaned_text

            # SAVE OCR RESULT
            if best_text != "":
                already_exists = False

                for plate_data in st.session_state.detected_plates:
                    if plate_data["Plate"] == best_text:
                        already_exists = True
                        break

                if not already_exists:
                    st.session_state.detected_plates.append({
                        "Plate": best_text,
                        "Confidence": round(best_score, 2)
                    })

                cv2.putText(
                    image,
                    best_text,
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    return image


# IMAGE MODE
if mode == "Image" and uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        1
    )

    processed_image = process_image(image)

    st.subheader("📷 Detection Result")

    st.image(
        cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )


# VIDEO MODE
elif mode == "Video" and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_image(frame)

        stframe.image(
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    cap.release()


# LIVE CAMERA MODE
elif mode == "Live Camera":
    st.subheader("🎥 Live Traffic Monitoring")

    run_camera = st.checkbox("Start Camera")

    stframe = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = cap.read()

            if not ret:
                st.error("Unable to access camera")
                break

            processed_frame = process_image(frame)

            stframe.image(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

        cap.release()


# ANALYTICS DASHBOARD
st.markdown("---")

col1, col2 = st.columns(2)

# VEHICLE COUNTS
with col1:
    st.subheader("🚘 Vehicle Counts")

    vehicle_df = pd.DataFrame(
        st.session_state.vehicle_counts.items(),
        columns=["Vehicle Type", "Count"]
    )

    st.dataframe(
        vehicle_df,
        use_container_width=True
    )


# DETECTED PLATES
with col2:
    st.subheader("🔤 Detected Plates")

    if len(st.session_state.detected_plates) > 0:
        plate_df = pd.DataFrame(
            st.session_state.detected_plates
        )

        st.dataframe(
            plate_df,
            use_container_width=True
        )

        csv = plate_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Plate Logs CSV",
            csv,
            "plate_logs.csv",
            "text/csv"
        )

    else:
        st.warning("No plates detected")

# import streamlit as st
# from ultralytics import YOLO
# import easyocr
# import cv2
# import numpy as np
# import tempfile
# import re

# # ==========================================
# # PAGE CONFIG
# # ==========================================

# st.set_page_config(
#     page_title="ANPR System",
#     page_icon="🚗",
#     layout="wide"
# )

# # ==========================================
# # SIDEBAR
# # ==========================================

# st.sidebar.title("ANPR System")
# st.sidebar.markdown("YOLOv8 + EasyOCR")
# st.sidebar.markdown("---")
# st.sidebar.markdown("Automatic Number Plate Recognition")

# # ==========================================
# # LOAD MODEL
# # ==========================================

# MODEL_PATH = "runs/detect/outputs/training/number_plate_detector/weights/best.pt"

# @st.cache_resource
# def load_model():
#     return YOLO(MODEL_PATH)

# @st.cache_resource
# def load_ocr():
#     return easyocr.Reader(['en'], gpu=True)

# model = load_model()
# reader = load_ocr()

# # ==========================================
# # OCR CLEANING
# # ==========================================

# def clean_text(text):

#     text = text.upper()

#     text = re.sub(r'[^A-Z0-9]', '', text)

#     return text

# # ==========================================
# # PREPROCESSING
# # ==========================================

# def preprocess_plate(crop):

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # enlarge image
#     gray = cv2.resize(
#         gray,
#         None,
#         fx=3,
#         fy=3,
#         interpolation=cv2.INTER_CUBIC
#     )

#     # denoise
#     gray = cv2.bilateralFilter(gray, 11, 17, 17)

#     # sharpen
#     kernel = np.array([
#         [-1,-1,-1],
#         [-1, 9,-1],
#         [-1,-1,-1]
#     ])

#     sharpened = cv2.filter2D(gray, -1, kernel)

#     # threshold
#     thresh = cv2.adaptiveThreshold(
#         sharpened,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11,
#         2
#     )

#     return thresh

# # ==========================================
# # STREAMLIT UI
# # ==========================================

# st.title("🚗 Automatic Number Plate Recognition (ANPR) System")

# st.write(
#     "This application detects vehicle number plates using YOLOv8 and extracts the plate text using EasyOCR."
# )

# uploaded_file = st.file_uploader(
#     "Upload Vehicle Image",
#     type=["jpg", "jpeg", "png"]
# )

# # ==========================================
# # PROCESS IMAGE
# # ==========================================

# if uploaded_file is not None:

#     # Save temporary image
#     tfile = tempfile.NamedTemporaryFile(delete=False)

#     tfile.write(uploaded_file.read())

#     # Read image
#     image = cv2.imread(tfile.name)

#     original = image.copy()

#     st.subheader("Uploaded Image")

#     st.image(
#         cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#         use_container_width=True
#     )

#     # ==========================================
#     # DETECTION
#     # ==========================================

#     results = model.predict(
#         source=image,
#         conf=0.40,
#         imgsz=960,
#         device=0
#     )

#     boxes = results[0].boxes

#     extracted_plates = []

#     # ==========================================
#     # LOOP THROUGH DETECTIONS
#     # ==========================================

#     for idx, box in enumerate(boxes):

#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         confidence = float(box.conf[0])

#         # crop plate
#         crop = original[y1:y2, x1:x2]

#         # preprocess
#         processed_crop = preprocess_plate(crop)

#         # OCR
#         ocr_results = reader.readtext(
#             processed_crop,
#             detail=0,
#             paragraph=False
#         )

#         detected_text = ""

#         if len(ocr_results) > 0:

#             detected_text = ocr_results[0]

#             detected_text = clean_text(detected_text)

#         # save results
#         extracted_plates.append({
#             "text": detected_text,
#             "confidence": confidence
#         })

#         # ==========================================
#         # DRAW BOUNDING BOX
#         # ==========================================

#         label = f"{detected_text} ({confidence:.2f})"

#         cv2.rectangle(
#             image,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             2
#         )

#         cv2.putText(
#             image,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 0, 0),
#             2
#         )

#     # ==========================================
#     # SHOW OUTPUT IMAGE
#     # ==========================================

#     st.subheader("Detection Results")

#     st.image(
#         cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#         use_container_width=True
#     )

#     # ==========================================
#     # SHOW EXTRACTED TEXT
#     # ==========================================

#     st.subheader("Extracted Plate Numbers")

#     if len(extracted_plates) > 0:

#         for idx, plate in enumerate(extracted_plates):

#             if plate["text"] != "":

#                 st.info(
#                     f"🚘 Plate {idx+1}: {plate['text']} | Confidence: {plate['confidence']:.2f}"
#                 )

#             else:

#                 st.warning(
#                     f"⚠️ Plate {idx+1}: OCR could not read text properly"
#                 )

#     else:

#         st.error("No number plates detected.")

# # ==========================================
# # FOOTER
# # ==========================================

# st.markdown("---")

# st.markdown(
#     "Built using YOLOv8, EasyOCR, OpenCV and Streamlit"
# )