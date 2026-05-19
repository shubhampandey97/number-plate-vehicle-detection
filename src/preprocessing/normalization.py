import cv2
import numpy as np



def min_max_normalization(image):

    normalized = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized



def z_score_normalization(image):

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    normalized = (image - mean) / (std + 1e-8)

    return normalized



def scale_pixels(image):

    image = image.astype(np.float32) / 255.0

    return image


if __name__ == "__main__":

    image = cv2.imread(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train\img_000000.jpg")

    normalized = min_max_normalization(image)

    cv2.imwrite("outputs/preprocessing/normalized_image.jpg", normalized)

    print("Normalization completed")
