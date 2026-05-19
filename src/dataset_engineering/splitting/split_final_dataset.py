import os
import random
import shutil
from pathlib import Path

# PATHS
IMAGE_DIR = "data/final_merged_datasets/images"
LABEL_DIR = "data/final_merged_datasets/labels"

OUTPUT_DIR = "data/final_merged_datasets"

# SPLIT RATIOS
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# CREATE FOLDERS
for split in ["train", "val", "test"]:

    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# GET IMAGE FILES
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

image_files = []

for ext in IMAGE_EXTENSIONS:
    image_files.extend(list(Path(IMAGE_DIR).glob(f"*{ext}")))

random.shuffle(image_files)

# SPLIT COUNTS
total = len(image_files)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# MOVE FILES
for split_name, files in splits.items():

    print(f"\nProcessing {split_name}: {len(files)} images")

    for image_path in files:

        label_path = Path(LABEL_DIR) / f"{image_path.stem}.txt"

        # Destination paths
        dst_image = Path(OUTPUT_DIR) / "images" / split_name / image_path.name
        dst_label = Path(OUTPUT_DIR) / "labels" / split_name / label_path.name

        shutil.move(str(image_path), str(dst_image))
        shutil.move(str(label_path), str(dst_label))

print("\n========== DATASET SPLIT COMPLETE ==========")