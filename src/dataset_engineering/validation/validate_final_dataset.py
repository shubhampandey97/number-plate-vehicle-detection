import os
import cv2

# SPLITS
SPLITS = ["train", "val", "test"]

# IMAGE EXTENSIONS
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# BASE DATASET PATH
BASE_DATASET_DIR = (
    r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets"
)

# OUTPUT DIRECTORY
OUTPUT_DIR = (r"D:\Guvi\Projects\major\number-plate-vehicle-detection\outputs\analysis\validation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_dataset_validation_report.txt"
)

# REPORT STORAGE
report_lines = []

header = "\n========== FINAL DATASET VALIDATION ==========\n"
print(header)

report_lines.append(header)

# PROCESS EACH SPLIT
for split in SPLITS:
    IMAGE_DIR = os.path.join(
        BASE_DATASET_DIR,
        "images",
        split
    )

    LABEL_DIR = os.path.join(
        BASE_DATASET_DIR,
        "labels",
        split
    )

    # CHECK FOLDERS
    if not os.path.exists(IMAGE_DIR):
        print(f"Image folder missing: {IMAGE_DIR}")

        continue

    if not os.path.exists(LABEL_DIR):
        print(f"Label folder missing: {LABEL_DIR}")

        continue

    image_names = set()
    label_names = set()

    corrupted_images = []
    empty_labels = []
    invalid_annotations = []

    # COLLECT IMAGE NAMES
    for file in os.listdir(IMAGE_DIR):
        ext = os.path.splitext(file)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            image_path = os.path.join(
                IMAGE_DIR,
                file
            )

            # corrupted image check
            image = cv2.imread(image_path)

            if image is None:
                corrupted_images.append(file)

                continue

            image_names.add(
                os.path.splitext(file)[0]
            )

    # COLLECT LABEL NAMES
    for file in os.listdir(LABEL_DIR):
        if file.endswith(".txt"):
            label_path = os.path.join(
                LABEL_DIR,
                file
            )

            label_names.add(
                os.path.splitext(file)[0]
            )

            # empty / invalid label check
            try:
                with open(label_path, "r") as f:

                    lines = f.readlines()

                # empty file
                if len(lines) == 0:
                    empty_labels.append(file)

                    continue

                # invalid YOLO annotation
                for line in lines:
                    parts = line.strip().split()

                    if len(parts) < 5:
                        invalid_annotations.append(file)

                        break

            except:

                invalid_annotations.append(file)

    # FIND UNMATCHED FILES
    unmatched_images = image_names - label_names

    unmatched_labels = label_names - image_names

    # PRINT RESULTS
    split_header = (f"\n========== {split.upper()} VALIDATION ==========\n")
    print(split_header)

    report_lines.append(split_header)

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

    # SAMPLE ISSUES
    print("\nSample Unmatched Images:")

    report_lines.append("\nSample Unmatched Images:")

    for name in list(unmatched_images)[:5]:
        print(name)

        report_lines.append(name)

    print("\nSample Unmatched Labels:")

    report_lines.append("\nSample Unmatched Labels:")

    for name in list(unmatched_labels)[:5]:
        print(name)

        report_lines.append(name)

# COMPLETE
footer = "\n========== VALIDATION COMPLETE =========="
print(footer)

report_lines.append(footer)

# SAVE REPORT
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nValidation report saved to:\n{OUTPUT_FILE}")