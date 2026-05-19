import os

IMAGE_DIR = "data/raw_datasets/images"
LABEL_DIR = "data/raw_datasets/labels"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

image_names = set()

# Collect image basenames
for file in os.listdir(IMAGE_DIR):

    ext = os.path.splitext(file)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        image_names.add(os.path.splitext(file)[0])

removed = 0

print("\n========== REMOVING UNMATCHED LABELS ==========\n")

# Remove labels without images
for label_file in os.listdir(LABEL_DIR):

    if not label_file.endswith(".txt"):
        continue

    label_name = os.path.splitext(label_file)[0]

    if label_name not in image_names:

        os.remove(os.path.join(LABEL_DIR, label_file))

        removed += 1
        print(f"Removed: {label_file}")

print("\n========== CLEANUP COMPLETE ==========")
print(f"Total Labels Removed: {removed}")