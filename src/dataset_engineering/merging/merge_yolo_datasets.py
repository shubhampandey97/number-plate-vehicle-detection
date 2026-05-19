import os
import shutil
from tqdm import tqdm

# Source datasets
SOURCE_DIRS = [
    "data/raw_datasets/dataset",
    "data/raw_datasets/data",
    "data/raw_datasets/Traffic Dataset"
]

# Final merged dataset
FINAL_DATASET = "data/final_merged_dataset"

# Create folders
os.makedirs(f"{FINAL_DATASET}/images/train", exist_ok=True)
os.makedirs(f"{FINAL_DATASET}/labels/train", exist_ok=True)

# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

counter = 0

print("\n========== MERGING DATASETS ==========\n")

for dataset_path in SOURCE_DIRS:

    print(f"Processing: {dataset_path}")

    for root, dirs, files in os.walk(dataset_path):

        for file in tqdm(files):

            ext = os.path.splitext(file)[1].lower()

            # Process images only
            if ext in IMAGE_EXTENSIONS:

                image_path = os.path.join(root, file)

                # Corresponding YOLO label
                label_name = os.path.splitext(file)[0] + ".txt"
                label_path = os.path.join(root, label_name)

                # Skip if label missing
                if not os.path.exists(label_path):
                    continue

                # New unique filename
                new_name = f"img_{counter:06d}"

                # Destination paths
                new_image_path = os.path.join(
                    FINAL_DATASET,
                    "images/train",
                    new_name + ext
                )

                new_label_path = os.path.join(
                    FINAL_DATASET,
                    "labels/train",
                    new_name + ".txt"
                )

                # Copy files
                shutil.copy(image_path, new_image_path)
                shutil.copy(label_path, new_label_path)

                counter += 1

print("\n========== MERGE COMPLETE ==========")
print(f"Total merged images: {counter}")