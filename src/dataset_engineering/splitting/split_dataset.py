import random
import shutil
from pathlib import Path

random.seed(42)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

SOURCE_IMAGES = Path("data/raw/custom_dataset/images")
SOURCE_LABELS = Path("data/raw/custom_dataset/labels")

DESTINATION = Path("data/final_dataset")

SPLITS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1
}


def create_folders():
    for split in SPLITS.keys():
        (DESTINATION / split / "images").mkdir(parents=True, exist_ok=True)
        (DESTINATION / split / "labels").mkdir(parents=True, exist_ok=True)


def get_all_images():
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(SOURCE_IMAGES.glob(f"*{ext}"))

    return images


def split_dataset(images):
    random.shuffle(images)

    total = len(images)

    train_end = int(total * SPLITS["train"])
    valid_end = train_end + int(total * SPLITS["valid"])

    return {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }


def copy_files(split_data):
    for split, images in split_data.items():

        for image_path in images:

            label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

            shutil.copy(
                image_path,
                DESTINATION / split / "images" / image_path.name
            )

            if label_path.exists():
                shutil.copy(
                    label_path,
                    DESTINATION / split / "labels" / label_path.name
                )


if __name__ == "__main__":

    create_folders()

    images = get_all_images()

    split_data = split_dataset(images)

    copy_files(split_data)

    print("Dataset split completed successfully.")