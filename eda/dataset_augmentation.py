import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

# Input dataset folder
image_folder = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train"
save_folder = r"D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\augmentation_effects"
os.makedirs(save_folder, exist_ok=True)

# Augmentation functions
def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    factor = random.uniform(0.5, 1.5)  # brightness scale
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def rotate(img):
    h, w = img.shape[:2]
    angle = random.choice([-15, -10, 10, 15])
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def simulate_rain(img):
    rain_layer = np.zeros_like(img, dtype=np.uint8)
    for _ in range(1000):
        x = random.randint(0, img.shape[1]-1)
        y = random.randint(0, img.shape[0]-1)
        rain_layer = cv2.line(rain_layer, (x, y), (x+1, y+10), (200,200,200), 1)
    return cv2.addWeighted(img, 0.8, rain_layer, 0.2, 0)

# Choose random images
image_files = random.sample(os.listdir(image_folder), min(5, len(os.listdir(image_folder))))

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    augmentations = {
        "Brightness": random_brightness(img),
        "Blur": add_blur(img),
        "Noise": add_noise(img),
        "Rotate": rotate(img),
        "Rain": simulate_rain(img),
    }

    # Plot original vs augmentations
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.title("Original")
    plt.axis("off")

    for i, (name, aug_img) in enumerate(augmentations.items(), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_folder, f"augmentation_{img_file}.png")
    plt.savefig(save_path)
    plt.close()

print(f"✅ Augmentation effects visualization saved in {save_folder}")
