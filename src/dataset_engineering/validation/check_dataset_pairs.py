import os
import glob
import cv2

# ROOT DATA DIRECTORY
DATA_DIR = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\raw_datasets"

# OUTPUT DIRECTORY
OUTPUT_DIR = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\outputs\dataset_engineering\validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_validation_report.txt"
)

# IMAGE EXTENSIONS
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# FIND ALL IMAGES & LABELS
image_files = []
label_files = []

for ext in IMAGE_EXTENSIONS:
    image_files.extend(
        glob.glob(
            os.path.join(DATA_DIR, "**", f"*{ext}"),
            recursive=True
        )
    )

label_files.extend(
    glob.glob(
        os.path.join(DATA_DIR, "**", "*.txt"),
        recursive=True
    )
)

# IMAGE & LABEL NAME SETS
image_names = set()
label_names = set()

# basename only
for image_path in image_files:
    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    image_names.add(image_name)

for label_path in label_files:
    label_name = os.path.splitext(
        os.path.basename(label_path)
    )[0]

    label_names.add(label_name)

# FIND UNMATCHED FILES
unmatched_images = image_names - label_names

unmatched_labels = label_names - image_names

# EXTRA VALIDATION
corrupted_images = []
empty_labels = []
invalid_annotations = []

# CORRUPTED IMAGES
for image_path in image_files:
    image = cv2.imread(image_path)

    if image is None:
        corrupted_images.append(image_path)

# EMPTY / INVALID LABELS
for label_path in label_files:
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()

        # empty file
        if len(lines) == 0:
            empty_labels.append(label_path)
            continue

        # invalid annotation
        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                invalid_annotations.append(label_path)
                break

    except:
        invalid_annotations.append(label_path)

# REPORT
report_lines = []

header = "\n========== DATASET VALIDATION ==========\n"

print(header)

report_lines.append(header)

stats = [
    f"Total Images: {len(image_names)}",
    f"Total Labels: {len(label_names)}",
    f"Unmatched Images: {len(unmatched_images)}",
    f"Unmatched Labels: {len(unmatched_labels)}",
    f"Corrupted Images: {len(corrupted_images)}",
    f"Empty Labels: {len(empty_labels)}",
    f"Invalid Annotations: {len(invalid_annotations)}"
]

for line in stats:
    print(line)

    report_lines.append(line)

# SAMPLE OUTPUTS
sample_header = "\n========== SAMPLE ISSUES ==========\n"
print(sample_header)

report_lines.append(sample_header)

# unmatched images
print("\nSample Unmatched Images:")

report_lines.append("\nSample Unmatched Images:")

for name in list(unmatched_images)[:10]:
    print(name)

    report_lines.append(name)

# unmatched labels
print("\nSample Unmatched Labels:")

report_lines.append("\nSample Unmatched Labels:")

for name in list(unmatched_labels)[:10]:
    print(name)

    report_lines.append(name)

# corrupted images
print("\nSample Corrupted Images:")

report_lines.append("\nSample Corrupted Images:")

for path in corrupted_images[:10]:
    print(path)

    report_lines.append(path)

# empty labels
print("\nSample Empty Labels:")

report_lines.append("\nSample Empty Labels:")

for path in empty_labels[:10]:
    print(path)

    report_lines.append(path)

# invalid annotations
print("\nSample Invalid Annotations:")

report_lines.append("\nSample Invalid Annotations:")

for path in invalid_annotations[:10]:
    print(path)

    report_lines.append(path)

footer = "\n========== VALIDATION COMPLETE =========="
print(footer)

report_lines.append(footer)

# SAVE REPORT
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nValidation report saved to:\n{OUTPUT_FILE}")