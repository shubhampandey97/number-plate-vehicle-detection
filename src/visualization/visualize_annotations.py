import cv2
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = {
    0: "Number plate"
}


def draw_yolo_boxes(image_path, label_path):

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:

        values = label.strip().split()

        class_id = int(values[0])

        x_center, y_center, bw, bh = map(float, values[1:])

        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("YOLO Annotation Visualization")
    plt.show()


if __name__ == "__main__":

    image_path = Path("data/final_dataset/train/images")
    label_path = Path("data/final_dataset/train/labels")

    images = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(list(image_path.glob(ext)))

    if len(images) == 0:
        raise ValueError("No images found in dataset path.")

    sample_image = images[0]

    sample_label = label_path / f"{sample_image.stem}.txt"

    draw_yolo_boxes(sample_image, sample_label)