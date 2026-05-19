from pathlib import Path

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

DATASET_PATH = Path("data/final_dataset")


def remove_unlabeled_images():

    splits = ['train', 'valid', 'test']

    total_removed = 0

    for split in splits:

        image_dir = DATASET_PATH / split / 'images'
        label_dir = DATASET_PATH / split / 'labels'

        images = []

        for ext in IMAGE_EXTENSIONS:
            images.extend(image_dir.glob(f"*{ext}"))

        removed = 0

        for image_path in images:

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():

                image_path.unlink()
                removed += 1

        total_removed += removed

        print(f"{split.upper()} -> Removed {removed} unlabeled images")

    print(f"\nTotal Removed: {total_removed}")


if __name__ == "__main__":
    remove_unlabeled_images()