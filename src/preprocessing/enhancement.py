import cv2
import numpy as np



def apply_clahe(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced



def histogram_equalization(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)

    return equalized



def sharpen_image(image):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened



def super_resolution_resize(image, scale=2):

    h, w = image.shape[:2]

    resized = cv2.resize(
        image,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )

    return resized


if __name__ == "__main__":

    image = cv2.imread(r"D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train\img_000000.jpg")

    enhanced = apply_clahe(image)

    cv2.imwrite("outputs/preprocessing/clahe_output.jpg", enhanced)

    print("Enhancement completed")
