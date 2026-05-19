import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import re
import os
import time


# =========================================================
# LOAD YOLO MODEL
# =========================================================

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


# CREATE OUTPUT DIRECTORY
os.makedirs("outputs", exist_ok=True)


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

output_video_path = "outputs/video_result1.mp4"

out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps_input,
    (frame_width, frame_height)
)


# FPS CALCULATION
prev_time = 0
frame_count = 0


# OCR CACHE
last_plate_text = ""
last_plate_conf = 0


# MAIN LOOP
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    print(f"\nProcessing Frame: {frame_count}")

    original_frame = frame.copy()

    # YOLO DETECTION
    results = model(frame)[0]

    # STORE BEST NUMBER PLATE
    best_plate_box = None
    best_plate_conf = 0

    # DRAW DETECTIONS
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
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{class_name} {conf:.2f}"

        cv2.putText(
            frame,
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

    # RUN OCR EVERY 5 FRAMES
    run_ocr = frame_count % 5 == 0

    # OCR ON BEST NUMBER PLATE ONLY
    if best_plate_box is not None and run_ocr:

        x1, y1, x2, y2 = map(
            int,
            best_plate_box.xyxy[0]
        )

        # CROP PLATE
        padding = 5

        plate_crop = original_frame[
            max(0, y1 - padding):min(original_frame.shape[0], y2 + padding),
            max(0, x1 - padding):min(original_frame.shape[1], x2 + padding)
        ]

        if plate_crop.size != 0:

            # RESIZE FOR BETTER OCR
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # GRAYSCALE
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

            # THRESHOLD
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

            # FIND BEST OCR RESULT
            for result in ocr_results:

                text = result[1]
                score = result[2]

                # Ignore weak OCR
                if score < 0.30:
                    continue

                # CLEAN TEXT
                cleaned_text = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )

                # Ignore short text
                if len(cleaned_text) < 4:
                    continue

                # KEEP BEST OCR
                if score > best_score:

                    best_score = score
                    best_text = cleaned_text

            # UPDATE OCR CACHE
            if best_text != "":

                last_plate_text = best_text
                last_plate_conf = best_score

                print(
                    f"Detected Plate: {best_text} | "
                    f"Confidence: {best_score:.2f}"
                )

    # DRAW LAST DETECTED OCR
    if best_plate_box is not None and last_plate_text != "":

        x1, y1, x2, y2 = map(
            int,
            best_plate_box.xyxy[0]
        )

        cv2.putText(
            frame,
            f"PLATE: {last_plate_text}",
            (x1, y2 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    # FPS DISPLAY
    current_time = time.time()

    fps = (
        1 / (current_time - prev_time)
        if prev_time != 0 else 0
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # WRITE FRAME TO VIDEO
    out.write(frame)

    print(f"FPS: {int(fps)}")

# RELEASE RESOURCES
cap.release()
out.release()

print(f"\nSaved Output Video: {output_video_path}")


# import cv2
# import easyocr
# from ultralytics import YOLO
# import numpy as np
# import re
# import os
# import time


# # LOAD YOLO MODEL
# model = YOLO(r"runs\detect\runs1\detect\traffic_object_detector-2\weights\best.pt")


# # LOAD OCR MODEL
# reader = easyocr.Reader(
#     ['en'],
#     gpu=True
# )


# # CLASS NAMES
# CLASS_NAMES = [
#     "Bike",
#     "Bus",
#     "Car",
#     "Number_plate",
#     "Person",
#     "Truck",
#     "Auto"
# ]


# # CREATE OUTPUT FOLDER
# os.makedirs("outputs", exist_ok=True)


# # VIDEO PATH
# video_path = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\raw_datasets\Traffic Dataset\images\Video11.mp4"

# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error opening video")
#     exit()


# # FPS CALCULATION
# prev_time = 0


# # MAIN LOOP
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     original_frame = frame.copy()

#     # YOLO DETECTION
#     results = model(frame)[0]

#     # FIND BEST NUMBER PLATE
#     best_plate_box = None
#     best_plate_conf = 0

#     # DRAW DETECTIONS
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])

#         class_name = CLASS_NAMES[cls_id]

#         x1, y1, x2, y2 = map(
#             int,
#             box.xyxy[0]
#         )

#         # DRAW BOX
#         color = (0, 255, 0)

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             color,
#             2
#         )

#         label = f"{class_name} {conf:.2f}"

#         cv2.putText(
#             frame,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             color,
#             2
#         )

#         # STORE BEST NUMBER PLATE
#         if class_name == "Number_plate":
#             if conf > best_plate_conf:
#                 best_plate_conf = conf
#                 best_plate_box = box

#     # OCR ON BEST PLATE
#     if best_plate_box is not None:
#         x1, y1, x2, y2 = map(
#             int,
#             best_plate_box.xyxy[0]
#         )

#         padding = 5

#         plate_crop = original_frame[
#             max(0, y1 - padding):min(original_frame.shape[0], y2 + padding),
#             max(0, x1 - padding):min(original_frame.shape[1], x2 + padding)
#         ]

#         if plate_crop.size != 0:
#             # PREPROCESSING
#             plate_crop = cv2.resize(
#                 plate_crop,
#                 None,
#                 fx=4,
#                 fy=4,
#                 interpolation=cv2.INTER_CUBIC
#             )

#             gray = cv2.cvtColor(
#                 plate_crop,
#                 cv2.COLOR_BGR2GRAY
#             )

#             gray = cv2.bilateralFilter(
#                 gray,
#                 11,
#                 17,
#                 17
#             )

#             _, thresh = cv2.threshold(
#                 gray,
#                 0,
#                 255,
#                 cv2.THRESH_BINARY + cv2.THRESH_OTSU
#             )

#             # OCR
#             ocr_results = reader.readtext(
#                 thresh,
#                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#                 paragraph=False,
#                 detail=1
#             )

#             best_text = ""
#             best_score = 0

#             for result in ocr_results:

#                 text = result[1]
#                 score = result[2]

#                 if score < 0.30:
#                     continue

#                 cleaned_text = re.sub(
#                     r'[^A-Z0-9]',
#                     '',
#                     text.upper()
#                 )

#                 if len(cleaned_text) < 4:
#                     continue

#                 if score > best_score:

#                     best_score = score
#                     best_text = cleaned_text

#             # DRAW OCR RESULT
#             if best_text != "":

#                 cv2.putText(
#                     frame,
#                     f"PLATE: {best_text}",
#                     (x1, y2 + 35),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 0, 0),
#                     2
#                 )

#     # FPS DISPLAY
#     current_time = time.time()

#     fps = 1 / (current_time - prev_time)

#     prev_time = current_time

#     cv2.putText(
#         frame,
#         f"FPS: {int(fps)}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2
#     )

#     # DISPLAY
#     cv2.imshow(
#         "Vehicle + Plate OCR Detection",
#         frame
#     )

#     # Press Q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# # CLEANUP
# cap.release()

# cv2.destroyAllWindows()