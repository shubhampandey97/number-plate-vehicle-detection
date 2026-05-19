from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def validate_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    # splits = ['train', 'valid', 'test']
    splits = ['train', 'val', 'test']

    for split in splits:
        # image_dir = dataset_path / split / 'images'
        # label_dir = dataset_path / split / 'labels'
        image_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split 

        print(f"\nChecking {split.upper()} dataset")

        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(image_dir.glob(f'*{ext}'))

        labels = list(label_dir.glob('*.txt'))

        print(f"Images Found : {len(images)}")
        print(f"Labels Found : {len(labels)}")

        missing_labels = []
        corrupted_images = []
        empty_labels = []

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels.append(image_path.name)

            try:
                img = Image.open(image_path)
                img.verify()
            except Exception:
                corrupted_images.append(image_path.name)

        for label_path in labels:
            if label_path.stat().st_size == 0:
                empty_labels.append(label_path.name)

        print(f"Missing Labels   : {len(missing_labels)}")
        print(f"Corrupted Images : {len(corrupted_images)}")
        print(f"Empty Labels     : {len(empty_labels)}")

        if missing_labels:
            print("\nSample Missing Labels:")
            print(missing_labels[:5])

        if corrupted_images:
            print("\nSample Corrupted Images:")
            print(corrupted_images[:5])

        if empty_labels:
            print("\nSample Empty Labels:")
            print(empty_labels[:5])


if __name__ == "__main__":
    validate_dataset("Traffic Dataset")