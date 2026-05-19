from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="data/raw/data",
    use_segments=False,
    use_keypoints=False
)

print("COCO to YOLO conversion complete.")