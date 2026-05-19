import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import re
import os


# LOAD YOLO MODEL
model = YOLO(r"runs\detect\runs1\detect\traffic_object_detector-2\weights\best.pt")

# LOAD OCR MODEL
reader = easyocr.Reader(
    ['en'],
    gpu=True
)


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


# MAIN FUNCTION
def detect_and_read(image_path):
    # CREATE OUTPUT DIRECTORY
    os.makedirs("outputs", exist_ok=True)

    # READ IMAGE
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read image")
        return

    # Keep clean copy for OCR
    original_image = image.copy()

    # YOLO DETECTION
    results = model(image)[0]

    # STORE BEST NUMBER PLATE
    best_plate_box = None
    best_plate_conf = 0

    # DRAW ALL DETECTIONS
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = CLASS_NAMES[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # DRAW DETECTION BOX
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

        # FIND BEST NUMBER PLATE
        if class_name == "Number_plate":

            if conf > best_plate_conf:

                best_plate_conf = conf
                best_plate_box = box

    # OCR ON BEST NUMBER PLATE ONLY
    if best_plate_box is not None:
        x1, y1, x2, y2 = map(
            int,
            best_plate_box.xyxy[0]
        )

        # CROP NUMBER PLATE
        padding = 5

        plate_crop = original_image[
            max(0, y1 - padding):min(original_image.shape[0], y2 + padding),
            max(0, x1 - padding):min(original_image.shape[1], x2 + padding)
        ]

        # Empty crop check
        if plate_crop.size != 0:
            # RESIZE FOR BETTER OCR
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # CONVERT TO GRAYSCALE
            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            # DENOISE
            gray = cv2.bilateralFilter(
                gray,
                11,
                17,
                17
            )

            # OTSU THRESHOLDING
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # SAVE THRESHOLD IMAGE
            cv2.imwrite(
                "outputs/threshold.jpg",
                thresh
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

            # FIND BEST OCR RESULT
            for result in ocr_results:

                text = result[1]
                score = result[2]

                # Ignore weak OCR
                if score < 0.30:
                    continue

                # Clean text
                cleaned_text = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )

                # Ignore tiny text
                if len(cleaned_text) < 4:
                    continue

                # Keep best OCR result
                if score > best_score:

                    best_score = score
                    best_text = cleaned_text

            # DRAW BEST OCR RESULT
            if best_text != "":
                print(
                    f"Detected Plate: {best_text} | "
                    f"Confidence: {best_score:.2f}"
                )

                cv2.putText(
                    image,
                    best_text,
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2
                )

    # SAVE OUTPUT IMAGE
    output_path = "outputs/ocr_result.jpg"

    cv2.imwrite(
        output_path,
        image
    )

    print(f"\nSaved Output Image: {output_path}")


# MAIN
if __name__ == "__main__":
    detect_and_read(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\test\img_000134.png")