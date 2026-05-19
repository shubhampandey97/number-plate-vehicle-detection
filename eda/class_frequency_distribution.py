import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

# Path to labels directory
LABEL_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\labels'
OUTPUT_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\class_frequency_distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Your class names
class_names = ['Bike', 'Bus', 'Car', 'Number plate', 'Person', 'Truck', 'Auto']

# Collect all label files (train, val, test)
label_files = glob.glob(os.path.join(LABEL_DIR, '**', '*.txt'), recursive=True)

class_counts = Counter()

# Parse label files
for lf in label_files:
    with open(lf, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])  # First number = class
            class_counts[class_id] += 1

print("Class Frequency Distribution:", class_counts)

# Plot if not empty
if class_counts:
    labels = [class_names[i] for i in class_counts.keys()]
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Class Frequency Distribution in Dataset")
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Save instead of show
    output_path = os.path.join(OUTPUT_DIR, "class_frequency_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Class frequency distribution plot saved to {output_path}")
else:
    print("⚠️ No class IDs found in dataset. Plot will be empty.")
