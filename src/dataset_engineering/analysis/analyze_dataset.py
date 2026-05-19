from pathlib import Path
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# CLASS NAMES
CLASS_NAMES = {
    0: "Bike",
    1: "Bus",
    2: "Car",
    3: "Number_plate",
    4: "Person",
    5: "Truck",
    6: "Auto"
}

# DATASET PATH
DATASET_PATH = Path(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets")

# OUTPUT DIRECTORY
OUTPUT_DIR = Path(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MAIN ANALYSIS FUNCTION
def analyze_dataset():
    class_counter = Counter()

    image_widths = []
    image_heights = []

    bbox_widths = []
    bbox_heights = []

    brightness_values = []

    corrupted_images = 0
    missing_labels = 0
    invalid_annotations = 0

    # correct split names
    splits = ['train', 'val', 'test']

    # PROCESS DATASET
    for split in splits:
        print(f"\nProcessing Split: {split}")

        image_dir = DATASET_PATH / "images" / split
        label_dir = DATASET_PATH / "labels" / split

        if not image_dir.exists():
            print(f"Image folder not found: {image_dir}")
            continue

        # all image formats
        images = []

        images.extend(list(image_dir.glob("*.jpg")))
        images.extend(list(image_dir.glob("*.jpeg")))
        images.extend(list(image_dir.glob("*.png")))

        print(f"Images Found: {len(images)}")

        for image_path in images:
            image = cv2.imread(str(image_path))

            # corrupted image
            if image is None:
                corrupted_images += 1
                continue

            h, w, _ = image.shape

            image_widths.append(w)
            image_heights.append(h)

            # brightness analysis
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            brightness_values.append(gray.mean())

            # label file
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels += 1
                continue

            with open(label_path, 'r') as f:
                labels = f.readlines()

            # PROCESS LABELS
            for label in labels:
                values = label.strip().split()

                # invalid annotation
                if len(values) < 5:
                    invalid_annotations += 1
                    continue

                try:
                    class_id = int(values[0])

                except:
                    invalid_annotations += 1
                    continue

                # invalid class
                if class_id not in CLASS_NAMES:
                    invalid_annotations += 1
                    continue

                class_counter[class_id] += 1

                # bbox format only
                try:
                    coords = list(map(float, values[1:]))

                    # YOLO bbox format
                    if len(coords) == 4:
                        x_center, y_center, bw, bh = coords

                        bbox_widths.append(bw)
                        bbox_heights.append(bh)

                except:
                    invalid_annotations += 1
                    continue

    # PRINT RESULTS
    print("\n====================================")
    print("CLASS DISTRIBUTION")
    print("====================================\n")

    for class_id, count in sorted(class_counter.items()):
        print(
            f"{CLASS_NAMES[class_id]} : {count}"
        )

    print("\n====================================")
    print("DATASET STATISTICS")
    print("====================================\n")

    print(f"Total Images Analyzed: {len(image_widths)}")
    print(f"Corrupted Images: {corrupted_images}")
    print(f"Missing Labels: {missing_labels}")
    print(f"Invalid Annotations: {invalid_annotations}")

    # SAVE CSV REPORT
    stats_df = pd.DataFrame({
        "Metric": [
            "Total Images",
            "Corrupted Images",
            "Missing Labels",
            "Invalid Annotations"
        ],
        "Value": [
            len(image_widths),
            corrupted_images,
            missing_labels,
            invalid_annotations
        ]
    })

    stats_df.to_csv(
        OUTPUT_DIR / "dataset_statistics.csv",
        index=False
    )

    # CLASS DISTRIBUTION
    plt.figure(figsize=(10, 6))

    labels = [
        CLASS_NAMES[i]
        for i in sorted(class_counter.keys())
    ]

    counts = [
        class_counter[i]
        for i in sorted(class_counter.keys())
    ]

    plt.bar(labels, counts)

    plt.title("Class Distribution")
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=300
    )

    plt.close()

    # BRIGHTNESS DISTRIBUTION
    plt.figure(figsize=(10, 6))

    plt.hist(
        brightness_values,
        bins=20
    )

    plt.title("Brightness Distribution")

    plt.xlabel("Brightness")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "brightness_distribution.png",
        dpi=300
    )

    plt.close()

    # BBOX WIDTH DISTRIBUTION
    if bbox_widths:
        plt.figure(figsize=(10, 6))

        plt.hist(
            bbox_widths,
            bins=20
        )

        plt.title(
            "Bounding Box Width Distribution"
        )

        plt.xlabel("Width")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "bbox_width_distribution.png",
            dpi=300
        )

        plt.close()

    # BBOX HEIGHT DISTRIBUTION
    if bbox_heights:
        plt.figure(figsize=(10, 6))

        plt.hist(
            bbox_heights,
            bins=20
        )

        plt.title(
            "Bounding Box Height Distribution"
        )

        plt.xlabel("Height")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "bbox_height_distribution.png",
            dpi=300
        )

        plt.close()

    print("\n====================================")
    print("DATASET ANALYSIS COMPLETED")
    print("====================================")

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


# MAIN
if __name__ == "__main__":
    analyze_dataset()