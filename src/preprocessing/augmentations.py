from ultralytics import YOLO



def get_training_config():

    config = {

        # Augmentations
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,

        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,

        "flipud": 0.0,
        "fliplr": 0.5,

        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,

        "close_mosaic": 10
    }

    return config


if __name__ == "__main__":

    config = get_training_config()

    print("Training Augmentations")

    for key, value in config.items():
        print(f"{key}: {value}")
