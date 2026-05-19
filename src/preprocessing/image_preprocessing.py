import cv2
import numpy as np


def resize_image(image, width=None, height=None):
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        ratio = width / w
        dim = (width, int(h * ratio))
    else:
        ratio = height / h
        dim = (int(w * ratio), height)

    resized = cv2.resize(
        image,
        dim,
        interpolation=cv2.INTER_AREA
    )

    return resized



def normalize_image(image):
    image = image.astype(np.float32) / 255.0

    return image



def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )



def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = 1 + (contrast / 100.0)

    adjusted = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )

    return adjusted



def sharpen_image(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


if __name__ == "__main__":
    image = cv2.imread(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train\img_000000.jpg")

    resized = resize_image(image, width=640)

    normalized = normalize_image(resized)

    denoised = denoise_image(resized)

    sharpened = sharpen_image(denoised)

    cv2.imwrite("outputs/preprocessing/preprocessed_image.jpg", sharpened)

    print("Image preprocessing completed")

