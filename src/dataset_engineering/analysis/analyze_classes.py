import os
import glob
from collections import Counter

# RAW DATASETS ROOT
RAW_DATASET_DIR = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\raw_datasets"

# OUTPUT DIRECTORY
OUTPUT_DIR = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\outputs\dataset_engineering\analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# OUTPUT FILE
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "class_analysis_report.txt"
)

# CLASS NAMES
class_names = {
    0: "Bike",
    1: "Bus",
    2: "Car",
    3: "Number_plate",
    4: "Person",
    5: "Truck",
    6: "Auto"
}

# FIND ALL LABEL FILES
label_files = glob.glob(
    os.path.join(RAW_DATASET_DIR, "**", "labels", "**", "*.txt"),
    recursive=True
)

print("\n========== ANALYZING RAW DATASET CLASSES ==========\n")

print(f"Total Label Files Found: {len(label_files)}\n")

# CLASS COUNTER
class_counter = Counter()

invalid_annotations = 0

# PROCESS LABEL FILES
for label_file in label_files:
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # invalid YOLO annotation
        if len(parts) < 5:
            invalid_annotations += 1
            continue

        try:
            class_id = int(parts[0])

        except:
            invalid_annotations += 1
            continue

        class_counter[class_id] += 1

# REPORT
report_lines = []

header = "\n========== RAW DATASET CLASS ANALYSIS ==========\n"
print(header)

report_lines.append(header)

for class_id, count in sorted(class_counter.items()):
    class_name = class_names.get(
        class_id,
        f"Unknown_{class_id}"
    )

    line = (
        f"Class {class_id} "
        f"({class_name}) : "
        f"{count} annotations"
    )

    print(line)

    report_lines.append(line)

invalid_line = (f"\nInvalid Annotations: {invalid_annotations}")
print(invalid_line)

report_lines.append(invalid_line)

footer = "\n========== ANALYSIS COMPLETE =========="

print(footer)

report_lines.append(footer)

# SAVE REPORT
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nReport saved to:\n{OUTPUT_FILE}")