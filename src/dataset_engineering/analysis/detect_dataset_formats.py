import os
from collections import defaultdict

# Root folder containing all datasets
ROOT_DIR = "data/raw_datasets"

OUTPUT_DIR = "outputs/dataset_engineering/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output report file
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_analysis_report.txt"
)
# Supported annotation formats
ANNOTATION_FORMATS = {
    ".txt": "YOLO",
    ".xml": "Pascal VOC",
    ".json": "COCO"
}

# Image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Store dataset information
dataset_info = defaultdict(lambda: {
    "images": 0,
    "annotations": 0,
    "formats": set(),
    "image_files": [],
    "annotation_files": []
})

# Store output text
report_lines = []

header = "\n========== DATASET ANALYSIS ==========\n"
print(header)
report_lines.append(header)

# Traverse all folders
for root, dirs, files in os.walk(ROOT_DIR):
    dataset_name = root.split(os.sep)[1] \
        if len(root.split(os.sep)) > 1 else root

    for file in files:
        file_path = os.path.join(root, file)

        ext = os.path.splitext(file)[1].lower()

        # Detect images
        if ext in IMAGE_EXTENSIONS:
            dataset_info[dataset_name]["images"] += 1
            dataset_info[dataset_name]["image_files"].append(file_path)

        # Detect annotation formats
        if ext in ANNOTATION_FORMATS:
            dataset_info[dataset_name]["annotations"] += 1
            dataset_info[dataset_name]["formats"].add(
                ANNOTATION_FORMATS[ext]
            )
            dataset_info[dataset_name]["annotation_files"].append(file_path)

# Print + Save dataset report
for dataset, info in dataset_info.items():
    lines = [
        f"Dataset: {dataset}",
        f"Images: {info['images']}",
        f"Annotations: {info['annotations']}",
        f"Formats Detected: {list(info['formats'])}"
    ]

    if info["images"] == 0:
        lines.append("WARNING: No images found")

    if info["annotations"] == 0:
        lines.append("WARNING: No annotations found")

    lines.append("-" * 50)

    # Print
    for line in lines:
        print(line)

    # Save
    report_lines.extend(lines)

footer = "\n========== ANALYSIS COMPLETE ==========\n"

print(footer)
report_lines.append(footer)

# Save report to txt file
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report_lines))

print(f"Report saved to: {OUTPUT_FILE}")