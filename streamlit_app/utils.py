import cv2
import easyocr
from ultralytics import YOLO

# =========================
# LOAD MODEL + OCR
# =========================

MODEL_PATH = "runs/detect/outputs/training/number_plate_detector/weights/best.pt"

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['en'], gpu=False)

# =========================
# CLEAN TEXT
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
# PREPROCESS PLATE
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

    # denoise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    return gray

# =========================
# MAIN DETECTION FUNCTION
# =========================

def detect_number_plate(image):

    original = image.copy()

    results = model.predict(
        source=image,
        conf=0.42,
        imgsz=960
    )

    boxes = results[0].boxes

    detected_plates = []

    for idx, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        # =========================
        # ADD PADDING
        # =========================

        padding = 10

        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(original.shape[1], x2 + padding)
        y2_p = min(original.shape[0], y2 + padding)

        # =========================
        # CROP PLATE
        # =========================

        crop = original[y1_p:y2_p, x1_p:x2_p]

        # =========================
        # PREPROCESS
        # =========================

        processed_crop = preprocess_plate(crop)

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

        # clean text
        plate_text = clean_text(plate_text)

        # =========================
        # SAVE RESULTS
        # =========================

        detected_plates.append({
            "text": plate_text,
            "confidence": confidence
        })

        # =========================
        # DRAW BOUNDING BOX
        # =========================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{plate_text} ({confidence:.2f})"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return image, detected_plates