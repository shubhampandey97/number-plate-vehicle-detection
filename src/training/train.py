from ultralytics import YOLO


def train_model():

    model = YOLO("yolov8l.pt")

    model.train(

        # =====================================================
        # DATA
        # =====================================================

        data="data/final_merged_datasets/data.yaml",

        # =====================================================
        # TRAINING
        # =====================================================

        epochs=150,

        imgsz=960,

        batch=24,

        device=0,

        workers=12,

        patience=40,

        pretrained=True,

        # =====================================================
        # OPTIMIZER
        # =====================================================

        optimizer="AdamW",

        lr0=0.0005,

        lrf=0.01,

        weight_decay=0.0005,

        momentum=0.937,

        # =====================================================
        # LOSS / TRAINING STABILITY
        # =====================================================

        cos_lr=True,

        amp=True,

        cache=True,

        deterministic=False,

        # =====================================================
        # AUGMENTATION
        # =====================================================

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,

        # =====================================================
        # SMALL OBJECT IMPROVEMENT
        # =====================================================

        close_mosaic=10,

        # =====================================================
        # VALIDATION / SAVING
        # =====================================================

        val=True,

        save=True,

        save_period=10,

        plots=True,

        verbose=True,

        # =====================================================
        # OUTPUT
        # =====================================================

        project="runs1/detect",

        name="traffic_object_detector"
    )


if __name__ == "__main__":
    train_model()






# from ultralytics import YOLO


# def train_model():

#     model = YOLO("yolov8m.pt")

#     model.train(

#         data="final_merged_datasets/data.yaml",

#         epochs=100,

#         imgsz=960,

#         batch=32,

#         device=0,

#         workers=8,

#         optimizer="AdamW",

#         lr0=0.001,

#         patience=30,

#         project="runs",

#         name="traffic_object_detector",

#         pretrained=True,

#         save=True,

#         save_period=10,

#         val=True,

#         verbose=True,

#         hsv_h=0.015,
#         hsv_s=0.7,
#         hsv_v=0.4,

#         degrees=10,
#         translate=0.1,
#         scale=0.5,
#         shear=2.0,

#         fliplr=0.5,

#         mosaic=1.0,
#         mixup=0.1
#     )


# if __name__ == "__main__":
#     train_model()



# from ultralytics import YOLO


# def train_model():

#     # load pretrained YOLOv8 nano
#     model = YOLO("yolov8n.pt")

#     model.train(

#         # dataset yaml
#         data="Traffic Dataset/data.yaml",

#         # training
#         epochs=100,
#         # imgsz=960,
#         # batch=8,
#         imgsz=640,
#         batch=4,

#         # hardware
#         device=0,
#         # workers=4,
#         workers=0,

#         # optimizer
#         optimizer="AdamW",
#         lr0=0.001,

#         # regularization
#         patience=30,

#         # saving
#         project="runs",
#         name="traffic_object_detector",
#         exist_ok=True,

#         # validation
#         val=True,

#         # # augmentation
#         # hsv_h=0.015,
#         # hsv_s=0.7,
#         # hsv_v=0.4,

#         # degrees=10,
#         # translate=0.1,
#         # scale=0.5,
#         # shear=2.0,

#         # fliplr=0.5,
#         # mosaic=1.0,
#         # mixup=0.1,

#         # lighter augmentations
#         hsv_h=0.015,
#         hsv_s=0.5,
#         hsv_v=0.4,

#         degrees=5,
#         translate=0.1,
#         scale=0.3,
#         shear=1.0,

#         fliplr=0.5,

#         mosaic=0.5,
#         mixup=0.0,

#         # save
#         save=True,
#         save_period=10,

#         # pretrained
#         pretrained=True,

#         # verbose
#         verbose=True
#     )


# if __name__ == "__main__":
#     train_model()



# # from ultralytics import YOLO


# # def train_model():

# #     model = YOLO("yolov8n.pt")

# #     model.train(
# #         data="data/final_dataset/data.yaml",
# #         epochs=50,
# #         imgsz=960,
# #         batch=8,
# #         device=0,
# #         workers=2,
# #         project="outputs/training",
# #         name="number_plate_detector",
# #         pretrained=True,
# #         optimizer="Adam",
# #         lr0=0.001,
# #         patience=20,
# #         save=True,
# #         save_period=5,
# #         val=True,
# #         verbose=True
# #     )


# # if __name__ == "__main__":
# #     train_model()