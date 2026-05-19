import cv2
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset path
IMAGE_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images'
OUTPUT_DIR = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\lighting_weather_report'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds for categorization
DARK_THRESHOLD = 60
BRIGHT_THRESHOLD = 190
LOW_CONTRAST_THRESHOLD = 25

def classify_lighting_weather(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    contrast = gray.std()

    if contrast < LOW_CONTRAST_THRESHOLD:
        return "Foggy/Low Contrast"
    elif brightness < DARK_THRESHOLD:
        return "Night/Dark"
    elif brightness > BRIGHT_THRESHOLD:
        return "Bright/Sunny"
    else:
        return "Daylight"

# Collect results
image_files = glob.glob(os.path.join(IMAGE_DIR, '*/*.jpg')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*/*.png'))

results = []
for img_file in image_files:
    img = cv2.imread(img_file)
    if img is None:
        continue
    category = classify_lighting_weather(img)
    results.append({"image": img_file, "category": category})

# Save CSV
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "lighting_weather_report.csv")
df.to_csv(csv_path, index=False)
print(f"Lighting/Weather analysis report saved: {csv_path}")

# Plot distribution
plt.figure(figsize=(6,5))

df['category'].value_counts().plot(
    kind='bar', 
    color='teal', 
    edgecolor='black'
)

plt.title("Lighting & Weather Condition Distribution")
plt.xlabel("Condition")
plt.ylabel("Number of Images")

plt.xticks(rotation=30)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "lighting_weather_distribution.png")
plt.savefig(plot_path)
plt.close()

print(f"📊 Distribution plot saved: {plot_path}")
