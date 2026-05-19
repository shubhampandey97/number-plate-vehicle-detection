import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import re
import os
import time
import pandas as pd
from datetime import datetime


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


# CREATE OUTPUT DIRECTORIES
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


# VIDEO PATH
video_path = (r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\raw_datasets\Traffic Dataset\images\Video11.mp4")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video -> {video_path}")
    exit()


# VIDEO PROPERTIES
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

if fps_input == 0:
    fps_input = 20


# VIDEO WRITER
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video_path = "outputs/traffic_analytics_output.mp4"

out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps_input,
    (frame_width, frame_height)
)


# TRACKING VARIABLES
tracked_plates = {}

unique_vehicle_ids = set()

vehicle_counts = {
    "Bike": 0,
    "Bus": 0,
    "Car": 0,
    "Truck": 0,
    "Auto": 0
}

counted_ids = set()

analytics_data = []


# FPS VARIABLES
prev_time = 0
frame_count = 0


# MAIN LOOP
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    original_frame = frame.copy()

    # YOLO TRACKING
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        imgsz=640
    )[0]

    # PROCESS DETECTIONS
    for box in results.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])

        cls_id = int(box.cls[0])

        conf = float(box.conf[0])

        class_name = CLASS_NAMES[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # UNIQUE VEHICLE COUNTING
        if class_name in vehicle_counts:
            if track_id not in counted_ids:
                vehicle_counts[class_name] += 1

                counted_ids.add(track_id)

        unique_vehicle_ids.add(track_id)

        # DRAW DETECTION
        color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"ID {track_id} | {class_name}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # OCR FOR NUMBER PLATES
        if class_name == "Number_plate":
            if conf < 0.7:
                continue

            # OCR ONLY ONCE
            if track_id not in tracked_plates:
                padding = 5

                plate_crop = original_frame[
                    max(0, y1 - padding):min(original_frame.shape[0], y2 + padding),
                    max(0, x1 - padding):min(original_frame.shape[1], x2 + padding)
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

                    if score < 0.30:
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

                    tracked_plates[track_id] = best_text

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    analytics_data.append({
                        "timestamp": timestamp,
                        "track_id": track_id,
                        "plate_number": best_text,
                        "confidence": round(best_score, 2)
                    })

                    print(
                        f"Track ID: {track_id} | "
                        f"Plate: {best_text}"
                    )

        # DISPLAY PLATE
        if track_id in tracked_plates:

            plate_text = tracked_plates[track_id]

            cv2.putText(
                frame,
                f"PLATE: {plate_text}",
                (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    # DISPLAY ANALYTICS
    analytics_y = 40

    for vehicle_type, count in vehicle_counts.items():
        cv2.putText(
            frame,
            f"{vehicle_type}: {count}",
            (20, analytics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        analytics_y += 30

    # FPS
    current_time = time.time()

    fps = (
        1 / (current_time - prev_time)
        if prev_time != 0 else 0
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, analytics_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # WRITE FRAME
    out.write(frame)


# SAVE ANALYTICS CSV
df = pd.DataFrame(analytics_data)

csv_path = "outputs/logs/plate_logs.csv"

df.to_csv(csv_path, index=False)


# RELEASE
cap.release()
out.release()


# FINAL SUMMARY

print("\n========== TRAFFIC ANALYTICS ==========")

for vehicle_type, count in vehicle_counts.items():
    print(f"{vehicle_type}: {count}")

print(f"\nUnique Vehicles Tracked: {len(unique_vehicle_ids)}")

print(f"CSV Log Saved: {csv_path}")

print(f"Output Video Saved: {output_video_path}")