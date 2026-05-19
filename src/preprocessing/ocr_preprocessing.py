import cv2
import numpy as np
import re



def preprocess_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return None

    # Resize
    plate_crop = cv2.resize(
        plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    # Sharpening
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    gray = cv2.filter2D(gray, -1, kernel)

    # Noise reduction
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # Thresholding
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh



def clean_ocr_text(text):

    cleaned_text = re.sub(
        r'[^A-Z0-9]',
        '',
        text.upper()
    )

    return cleaned_text



def validate_plate(text, min_length=4):
    if len(text) < min_length:
        return False

    return True


if __name__ == "__main__":
    image = cv2.imread(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train\img_000000.jpg")

    processed = preprocess_plate(image)

    if processed is not None:
        cv2.imwrite("outputs/preprocessing/processed_plate.jpg", processed)

    print("OCR preprocessing completed")
