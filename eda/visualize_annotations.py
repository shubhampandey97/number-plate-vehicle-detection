import cv2
import glob
import os
import numpy as np

# CLASS NAMES
class_names = [
    'Bike',
    'Bus',
    'Car',
    'Number_plate',
    'Person',
    'Truck',
    'Auto'
]

# COLORS
object_colors = [
    (0, 0, 255),      # Bike
    (0, 255, 0),      # Bus
    (255, 0, 0),      # Car
    (255, 0, 255),    # Number Plate
    (255, 255, 0),    # Person
    (0, 255, 255),    # Truck
    (0, 165, 255)     # Auto
]

# PATHS
IMAGE_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images'

LABEL_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\labels'

OUTPUT_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\visualization'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CLASS HISTOGRAM
hist = [0] * len(class_names)

# DRAW FUNCTION
def draw_annotations(image, polygons, labels):
    output_image = image.copy()

    for idx, polygon in enumerate(polygons):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

        class_id = int(labels[idx])

        # safety check
        if class_id >= len(class_names):
            continue

        color = object_colors[class_id]

        # draw polygon / bbox
        cv2.polylines(
            output_image,
            [pts],
            isClosed=True,
            color=color,
            thickness=2
        )

        # label position
        x_min = np.min(pts[:, 0, 0])
        y_min = np.min(pts[:, 0, 1])

        cv2.putText(
            output_image,
            class_names[class_id],
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return output_image


# LOAD IMAGE FILES
image_files = []

image_files.extend(
    glob.glob(os.path.join(IMAGE_DIR, '*/*.jpg'))
)

image_files.extend(
    glob.glob(os.path.join(IMAGE_DIR, '*/*.jpeg'))
)

image_files.extend(
    glob.glob(os.path.join(IMAGE_DIR, '*/*.png'))
)

print(f"\nTotal Images Found: {len(image_files)}\n")

# PROCESS EACH IMAGE
for image_file in image_files:
    print(f"Processing: {image_file}")

    image = cv2.imread(image_file)

    # corrupted image check
    if image is None:
        print(f"Could not read image: {image_file}")
        continue

    # --------------------------------------
    # FIND LABEL FILE
    # --------------------------------------
    relative_path = os.path.relpath(image_file, IMAGE_DIR)

    label_file = os.path.join(
        LABEL_DIR,
        os.path.splitext(relative_path)[0] + '.txt'
    )

    polygons = []
    labels = []

    # READ LABELS
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        for line_num, label_line in enumerate(label_lines):
            label_line = label_line.strip()

            if not label_line:
                continue

            parts = label_line.split()

            # invalid annotation
            if len(parts) < 5:
                print(f"Skipping invalid annotation in {label_file}")
                continue

            # ----------------------------------
            # CLASS ID
            # ----------------------------------
            try:
                class_id = int(parts[0])

            except:
                print(f"Invalid class ID in {label_file}")
                continue

            if class_id >= len(class_names):
                print(f"Class ID out of range in {label_file}")
                continue

            # COORDINATES
            try:
                coords = list(map(float, parts[1:]))

            except:
                print(f"Invalid coordinates in {label_file}")
                continue

            h, w, _ = image.shape

            polygon_pts = []

            # YOLO BBOX FORMAT
            # class x_center y_center width height
            if len(coords) == 4:

                x_center, y_center, width, height = coords

                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)

                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # clamp
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                polygon_pts = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2)
                ]

            # POLYGON / SEGMENTATION FORMAT
            elif len(coords) >= 8 and len(coords) % 2 == 0:
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)

                    # clamp
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))

                    polygon_pts.append((x, y))

            else:
                print(f"Skipping invalid annotation in {label_file}")
                continue

            polygons.append(polygon_pts)
            labels.append(class_id)

            hist[class_id] += 1

    else:
        print(f"Label file missing: {label_file}")

    # DRAW ANNOTATIONS
    result_image = draw_annotations(
        image,
        polygons,
        labels
    )

    # OUTPUT FILE
    output_file = os.path.join(
        OUTPUT_DIR,
        relative_path
    )

    output_file = os.path.splitext(output_file)[0] + '.png'

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    cv2.imwrite(output_file, result_image)

# FINAL CLASS DISTRIBUTION
print("\n===================================")
print("CLASS DISTRIBUTION")
print("===================================\n")

for idx, class_name in enumerate(class_names):
    print(f"{class_name}: {hist[idx]}")

print("\nVisualization completed successfully.")
print(f"\nResults saved in:\n{OUTPUT_DIR}")