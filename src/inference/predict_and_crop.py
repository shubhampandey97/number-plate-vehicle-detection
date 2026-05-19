import numpy as np
from ultralytics import YOLO
import easyocr
import cv2
from pathlib import Path

# =========================
# PATHS
# =========================

MODEL_PATH = r"runs\detect\runs1\detect\traffic_object_detector-2\weights\best.pt"

INPUT_DIR = Path("inference/input")
OUTPUT_DIR = Path("inference/output")
CROP_DIR = Path("inference/crops")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD MODEL + OCR
# =========================

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['en'], gpu=True)

# =========================
# OCR CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.upper()

    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    cleaned = ""

    for char in text:
        if char in allowed:
            cleaned += char

    return cleaned

# =========================
# PREPROCESSING
# =========================

def preprocess_plate(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # upscale image
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # denoise while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    return gray

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():

    image_paths = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(INPUT_DIR.glob(ext))

    if len(image_paths) == 0:
        print("No input images found.")
        return

    for image_path in image_paths:

        print(f"\nProcessing: {image_path.name}")

        image = cv2.imread(str(image_path))

        original = image.copy()

        # =========================
        # YOLO PREDICTION
        # =========================

        results = model.predict(
            source=image,
            conf=0.42,
            imgsz=960,
            device=0
        )

        boxes = results[0].boxes

        for idx, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            # =========================
            # CROP PLATE
            # =========================

            # add padding
            padding = 10

            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(original.shape[1], x2 + padding)
            y2_p = min(original.shape[0], y2 + padding)

            crop = original[y1_p:y2_p, x1_p:x2_p]

            crop_filename = f"{image_path.stem}_crop_{idx}.jpg"

            crop_path = CROP_DIR / crop_filename

            cv2.imwrite(str(crop_path), crop)

            # =========================
            # PREPROCESS
            # =========================

            processed_crop = preprocess_plate(crop)

            processed_crop_path = CROP_DIR / f"{image_path.stem}_processed_{idx}.jpg"

            cv2.imwrite(str(processed_crop_path), processed_crop)

            # =========================
            # OCR
            # =========================

            ocr_results = reader.readtext(
                processed_crop,
                detail=0,
                paragraph=True
            )

            plate_text = ""

            if len(ocr_results) > 0:
                plate_text = ocr_results[0]

            # clean OCR text
            plate_text = clean_text(plate_text)

            print(f"Detected Plate: {plate_text}")

            # =========================
            # DRAW RESULTS
            # =========================

            label = f"{plate_text} ({confidence:.2f})"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        # =========================
        # SAVE FINAL OUTPUT
        # =========================

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(str(output_path), image)

        print(f"Saved Output: {output_path}")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_pipeline()