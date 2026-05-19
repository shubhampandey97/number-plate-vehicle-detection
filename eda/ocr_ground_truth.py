import os
from pathlib import Path

# Paths
DATASET_DIR = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets"
OUTPUT_FILE = os.path.join(DATASET_DIR, "annotations", "ocr_gt.txt")

# Create annotations directory if not exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Collect image paths
image_extensions = [".jpg", ".jpeg", ".png"]
image_list = []

for subset in ["train", "val", "test"]:
    subset_dir = Path(DATASET_DIR) / "images" / subset
    if subset_dir.exists():
        for file in subset_dir.iterdir():
            if file.suffix.lower() in image_extensions:
                relative_path = f"{subset}/{file.name}"
                image_list.append(relative_path)

# Write to ocr_gt.txt
with open(OUTPUT_FILE, "w") as f:
    for img in sorted(image_list):
        f.write(f"{img} \n")   # leave plate text blank for now

print(f"✅ Ground truth file created at: {OUTPUT_FILE}")
print(f"🖼️ Total images listed: {len(image_list)}")
