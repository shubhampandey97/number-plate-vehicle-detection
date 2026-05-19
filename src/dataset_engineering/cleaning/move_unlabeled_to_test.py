import os
import shutil

IMAGE_DIR = "data/raw_datasets/images"
LABEL_DIR = "data/raw_datasets/labels"

DEST_DIR = "data/final_merged_datasets/images/test"

os.makedirs(DEST_DIR, exist_ok=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

label_names = set()

# Collect label names
for file in os.listdir(LABEL_DIR):

    if file.endswith(".txt"):
        label_names.add(os.path.splitext(file)[0])

moved = 0

print("\n========== MOVING UNMATCHED IMAGES ==========\n")

for image_file in os.listdir(IMAGE_DIR):

    ext = os.path.splitext(image_file)[1].lower()

    if ext not in IMAGE_EXTENSIONS:
        continue

    image_name = os.path.splitext(image_file)[0]

    if image_name not in label_names:

        src_path = os.path.join(IMAGE_DIR, image_file)
        dst_path = os.path.join(DEST_DIR, image_file)

        shutil.move(src_path, dst_path)

        moved += 1
        print(f"Moved: {image_file}")

print("\n========== COMPLETE ==========")
print(f"Total Images Moved: {moved}")