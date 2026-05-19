import cv2
import glob
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset paths
IMAGE_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images'
LABEL_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\labels'

# Thresholds
BLUR_THRESHOLD = 100          # Lower than this → blurry
CONTRAST_THRESHOLD = 30       # Lower than this → occluded/low contrast
DARK_THRESHOLD = 50           # Gray mean < → too dark
BRIGHT_THRESHOLD = 200        # Gray mean > → too bright

# Output
OUTPUT_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\image_quality_analysis'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "image_quality_report.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions
def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < BLUR_THRESHOLD, lap_var

def detect_occlusion_and_lighting(plate_roi):
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    mean_brightness = gray.mean()
    occluded = contrast < CONTRAST_THRESHOLD

    if mean_brightness < DARK_THRESHOLD:
        lighting = "Too Dark"
    elif mean_brightness > BRIGHT_THRESHOLD:
        lighting = "Too Bright"
    else:
        lighting = "Good"

    return occluded, contrast, lighting

# Loop through all images
image_files = glob.glob(os.path.join(IMAGE_DIR, '*/*.jpg')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*/*.png'))

report_data = []

for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        continue

    # Default values
    blur_status, lap_var = detect_blur(image)
    occlusion_status = False
    contrast_val = 0
    lighting_status = "Unknown"

    # Corresponding label file
    label_file = image_file.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                label = int(parts[0])
                # Only check number plates (assume class index 3)
                if label != 3:
                    continue

                if len(parts) == 5:  # YOLO v5 format
                    x_c, y_c, w_box, h_box = map(float, parts[1:])
                    h_img, w_img, _ = image.shape
                    x1 = int((x_c - w_box/2) * w_img)
                    y1 = int((y_c - h_box/2) * h_img)
                    x2 = int((x_c + w_box/2) * w_img)
                    y2 = int((y_c + h_box/2) * h_img)
                    plate_roi = image[y1:y2, x1:x2]

                elif len(parts) == 9:  # Polygon 8-value + class
                    bbox_values = list(map(float, parts[1:9]))
                    h_img, w_img, _ = image.shape
                    xs = [int(bbox_values[i]*w_img) for i in range(0,8,2)]
                    ys = [int(bbox_values[i]*h_img) for i in range(1,8,2)]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    plate_roi = image[y1:y2, x1:x2]

                else:
                    print(f"Skipping invalid line in {label_file}: {line}")
                    continue

                # Check occlusion and lighting
                occlusion_status, contrast_val, lighting_status = detect_occlusion_and_lighting(plate_roi)
                break  # only check first number plate per image

    report_data.append({
        "image": image_file,
        "blur": "Yes" if blur_status else "No",
        "laplacian_var": f"{lap_var:.2f}",
        "occlusion_low_contrast": "Yes" if occlusion_status else "No",
        "contrast": f"{contrast_val:.2f}",
        "lighting": lighting_status
    })

# Write CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ["image", "blur", "laplacian_var", "occlusion_low_contrast", "contrast", "lighting"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in report_data:
        writer.writerow(row)

print(f"Image quality analysis completed. Report saved to {OUTPUT_CSV}")

# Generate Plots
df = pd.DataFrame(report_data)

# Blur Pie Chart
plt.figure(figsize=(5,5))

df['blur'].value_counts().plot(
    kind='pie', 
    autopct='%1.1f%%', 
    colors=['lightcoral','lightgreen']
)

plt.title("Blurred vs Clear Images")
plt.ylabel("")

plt.savefig(os.path.join(OUTPUT_DIR, "blur_distribution.png"))
plt.close()


# Lighting Bar Chart
plt.figure(figsize=(6,4))

df['lighting'].value_counts().plot(
    kind='bar', 
    color='skyblue', 
    edgecolor='black'
)

plt.title("Lighting Condition Distribution")
plt.xlabel("Lighting Condition")
plt.ylabel("Number of Images")

# Fix label cutting
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "lighting_distribution.png"),
    bbox_inches='tight'
)
plt.close()


# Occlusion Pie Chart
plt.figure(figsize=(5,5))

df['occlusion_low_contrast'].value_counts().plot(
    kind='pie', 
    autopct='%1.1f%%', 
    colors=['orange','lightblue']
)

plt.title("Low Contrast (Occluded) vs Clear")
plt.ylabel("")

plt.savefig(os.path.join(OUTPUT_DIR, "occlusion_distribution.png"))
plt.close()


# Laplacian Variance Histogram
plt.figure(figsize=(6,4))
df['laplacian_var'] = df['laplacian_var'].astype(float)
plt.hist(
    df['laplacian_var'], 
    bins=30, 
    color='purple', 
    edgecolor='black'
)

plt.title("Laplacian Variance Distribution (Blur Measure)")
plt.xlabel("Variance of Laplacian")
plt.ylabel("Number of Images")

plt.savefig(os.path.join(OUTPUT_DIR, "laplacian_variance_histogram.png"))
plt.close()

print(f"📊 Plots saved to {OUTPUT_DIR}")