from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = r"runs\detect\runs1\detect\traffic_object_detector-2\weights\best.pt"

INPUT_DIR = Path("inference/input")
OUTPUT_DIR = Path("inference/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_inference():

    model = YOLO(MODEL_PATH)

    image_paths = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(INPUT_DIR.glob(ext))

    if len(image_paths) == 0:
        print("No input images found.")
        return

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        results = model.predict(
            source=image,
            conf=0.42,
            imgsz=960,
            device=0
        )

        annotated_image = results[0].plot()

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(str(output_path), annotated_image)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_inference()