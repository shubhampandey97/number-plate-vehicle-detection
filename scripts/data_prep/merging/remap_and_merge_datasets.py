import os
import shutil
from pathlib import Path

# FINAL CLASS IDS
FINAL_CLASSES = {
    "Bike": 0,
    "Bus": 1,
    "Car": 2,
    "Number_plate": 3,
    "Person": 4,
    "Truck": 5,
    "Auto": 6
}

# DATASET CLASS MAPPINGS
DATASET_MAPPINGS = {

    "dataset": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    },

    "roboflow_dataset": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    },

    "Traffic Dataset": {
        0: 2,
        1: 3,
        2: 3,
        3: 0,
        4: 6,
        5: 1,
        6: 5
    },

    "custom_dataset": {
        0: 3
    }
}

# PATHS
RAW_DATASETS = "data/raw_datasets"

FINAL_IMAGE_DIR = "data/final_merged_datasets/images"
FINAL_LABEL_DIR = "data/final_merged_datasets/labels"

os.makedirs(FINAL_IMAGE_DIR, exist_ok=True)
os.makedirs(FINAL_LABEL_DIR, exist_ok=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

counter = 0

print("\n========== REMAPPING + MERGING ==========\n")

# PROCESS EACH DATASET
for dataset_name, mapping in DATASET_MAPPINGS.items():
    print(f"\nProcessing: {dataset_name}")

    dataset_path = Path(RAW_DATASETS) / dataset_name

    # Find ALL images recursively
    image_files = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(dataset_path.rglob(f"*{ext}"))

    print(f"Found {len(image_files)} images")

    for image_path in image_files:

        image_stem = image_path.stem

        # Find matching label anywhere inside dataset
        label_candidates = list(dataset_path.rglob(f"{image_stem}.txt"))

        if len(label_candidates) == 0:
            continue

        label_path = label_candidates[0]

        # New filenames
        new_name = f"img_{counter:06d}"

        new_image_path = Path(FINAL_IMAGE_DIR) / f"{new_name}{image_path.suffix}"
        new_label_path = Path(FINAL_LABEL_DIR) / f"{new_name}.txt"

        # Copy image
        shutil.copy(image_path, new_image_path)

        # Remap labels
        new_lines = []

        with open(label_path, "r") as f:

            lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                old_class = int(parts[0])

                if old_class not in mapping:
                    continue

                new_class = mapping[old_class]

                parts[0] = str(new_class)

                new_lines.append(" ".join(parts))

        # Skip empty labels
        if len(new_lines) == 0:
            continue

        # Save remapped label
        with open(new_label_path, "w") as f:
            f.write("\n".join(new_lines))

        counter += 1

print("\n========== COMPLETE ==========")
print(f"Total Images Processed: {counter}")