# 🚗 AI Traffic Monitoring & Automatic Number Plate Recognition (ANPR) System

An end-to-end **AI-powered Traffic Monitoring and Automatic Number Plate Recognition (ANPR)** system built using **YOLOv8, EasyOCR, ByteTrack, OpenCV, and Streamlit**.

This project performs:

- 🚘 Multi-class vehicle detection
- 🔢 Number plate detection
- 🧠 OCR-based plate recognition
- 🎯 Real-time vehicle tracking
- 📊 Traffic analytics
- 🎥 Video processing
- 📷 Live camera monitoring
- 🌐 Interactive Streamlit dashboard

---

# 📌 Project Overview

This project is a complete production-style **Computer Vision and Deep Learning pipeline** for intelligent traffic monitoring and vehicle number plate recognition.

The system can:
- Detect multiple vehicles in real time
- Detect and recognize vehicle number plates
- Track vehicles across frames
- Perform OCR using EasyOCR
- Generate traffic analytics
- Export detected plate logs
- Process images, videos, and live camera feeds

The project demonstrates:
- Deep Learning
- Object Detection
- OCR
- Object Tracking
- Real-time Video Processing
- AI Dashboard Development

---

# 🎯 Objectives

The main objectives of this project are:

- Detect vehicles and number plates accurately
- Handle small and distant objects
- Extract plate text automatically using OCR
- Track vehicles across video frames
- Build a production-style AI traffic monitoring pipeline
- Create a portfolio-ready deep learning project

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| YOLOv8 | Object Detection |
| EasyOCR | Text Recognition |
| ByteTrack | Vehicle Tracking |
| OpenCV | Image & Video Processing |
| PyTorch | Deep Learning Framework |
| CUDA | GPU Acceleration |
| Streamlit | Interactive Dashboard |
| Pandas | Analytics & CSV Export |
| NumPy | Numerical Operations |

---

# 🏗️ System Architecture

```text
Input Image / Video / Live Camera
                ↓
        YOLOv8 Detection
                ↓
      Vehicle & Plate Detection
                ↓
          ByteTrack Tracking
                ↓
        Plate Region Cropping
                ↓
         Image Preprocessing
                ↓
          EasyOCR Recognition
                ↓
        Traffic Analytics
                ↓
        Streamlit Dashboard
```

---

# ✨ Features

# ✅ Multi-Class Vehicle Detection

Detects:
- Car
- Bus
- Truck
- Bike
- Auto
- Person
- Number Plate

---

# ✅ Automatic Number Plate Recognition (ANPR)

The OCR pipeline:
1. Detects plates using YOLOv8
2. Crops plate regions
3. Applies preprocessing
4. Extracts text using EasyOCR

---

# ✅ Real-Time Vehicle Tracking

Integrated with **ByteTrack** for:
- Persistent vehicle IDs
- Vehicle movement tracking
- Duplicate OCR prevention

---

# ✅ Traffic Analytics

Provides:
- Vehicle counting
- Unique vehicle tracking
- Plate logging
- CSV export
- Analytics visualization

---

# ✅ Streamlit Dashboard

Interactive dashboard with:
- Image upload
- Video upload
- Live webcam monitoring
- Real-time analytics
- CSV download support

---

# 📂 Project Structure

```text
number-plate-vehicle-detection/
│
├── data/
│   ├── raw_datasets/
│   └── final_merged_datasets/
│
├── eda/
│   ├── dataset_stats.py
│   ├── dataset_augmentation.py
│   ├── image_quality_analysis.py
│   ├── lighting_weather_analysis.py
│   └── visualize_annotations.py
│
├── outputs/
│   ├── logs/
│   ├── ocr_result.jpg
│   ├── tracked_output.mp4
│   └── traffic_analytics_output.mp4
│
├── runs/
│
├── src/
│   │
│   ├── dataset_engineering/
│   │   │
│   │   ├── analysis/
│   │   │   ├── analyze_dataset.py
│   │   │   ├── analyze_classes.py
│   │   │   └── detect_dataset_formats.py
│   │   │
│   │   ├── cleaning/
│   │   │   ├── remove_unmatched_labels.py
│   │   │   ├── remove_unlabeled_images.py
│   │   │   └── move_unlabeled_to_test.py
│   │   │
│   │   ├── conversion/
│   │   │   └── convert_coco_to_yolo.py
│   │   │
│   │   ├── merging/
│   │   │   ├── merge_yolo_datasets.py
│   │   │   └── remap_and_merge_datasets.py
│   │   │
│   │   ├── splitting/
│   │   │   ├── split_dataset.py
│   │   │   └── split_final_dataset.py
│   │   │
│   │   └── validation/
│   │       ├── validate_dataset.py
│   │       ├── validate_final_dataset.py
│   │       └── check_dataset_pairs.py
│   │
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── inference/
│   │   ├── detect_image.py
│   │   └── detect_video.py
│   │
│   ├── ocr/
│   │   ├── plate_ocr.py
│   │   └── video_ocr.py
│   │
│   ├── tracking/
│   │   └── vehicle_tracking.py
│   │
│   ├── analytics/
│       └── traffic_analytics.py
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

---

# 📊 Dataset Engineering Pipeline

A professional dataset engineering workflow was followed before training.

# ✅ Dataset Validation

Implemented scripts to:
- Detect missing labels
- Detect corrupted images
- Detect empty annotations
- Validate YOLO annotation structure

---

# ✅ Dataset Cleaning

- Removed unlabeled images
- Standardized YOLO dataset format
- Organized train/validation/test splits

---

# ✅ Annotation Visualization

Bounding boxes were visualized to verify:
- Correct localization
- Proper label alignment
- Annotation consistency

---

# ✅ Exploratory Data Analysis (EDA)

Performed:
- Class distribution analysis
- Bounding box analysis
- Brightness analysis
- Image quality analysis
- Lighting/weather analysis

---

# 📊 Dataset Details

The model was trained on a custom merged traffic dataset containing:

- Cars
- Trucks
- Buses
- Bikes
- Autos
- Number Plates
- Blur Number Plates

## Dataset Statistics

| Class | Count |
|---|---|
| Car | 4116 |
| Number Plate | 479 |
| Blur Number Plate | 1149 |
| Two Wheeler | 1598 |
| Auto | 1072 |
| Bus | 305 |
| Truck | 435 |

---

# 🧪 Model Training

## Model Used

```python
YOLOv8l
```

---

## Training Techniques

The model was trained using:
- Transfer Learning
- AdamW Optimizer
- Mixed Precision Training (AMP)
- Mosaic Augmentation
- MixUp Augmentation
- HSV Augmentation
- Multi-class Object Detection

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 150 |
| Image Size | 960 |
| Batch Size | 24 |
| Optimizer | AdamW |
| Framework | Ultralytics YOLOv8 |

---

# ⚡ GPU Training

Training was performed using GPU acceleration with:
- PyTorch
- CUDA
- NVIDIA GPU

The project successfully utilized GPU acceleration for:
- Faster training
- Real-time inference
- OCR processing

---

# 🔍 OCR Pipeline

The OCR system performs:
1. Plate cropping
2. Image resizing
3. Bilateral filtering
4. Thresholding
5. OCR text extraction

---

## OCR Enhancements

Implemented preprocessing techniques:
- Grayscale conversion
- Image upscaling
- Bilateral filtering
- Otsu thresholding
- OCR text cleaning

---

# 🎯 Tracking Pipeline

Vehicle tracking is implemented using:
## ByteTrack

The tracking system:
- Assigns unique IDs
- Prevents duplicate OCR
- Tracks vehicles across frames
- Maintains persistent vehicle identity

---

# 📈 Traffic Analytics

The analytics system provides:
- Vehicle counting
- Plate logging
- Unique vehicle tracking
- CSV export
- Real-time analytics visualization

---

# 🚀 Streamlit Dashboard

Run the interactive dashboard:

```bash
streamlit run streamlit_app/app.py
```

The dashboard supports:
- Image upload
- Video upload
- Live webcam feed
- OCR visualization
- Analytics display

---

# 🎥 Video OCR

Run real-time video inference:

```bash
python src/ocr/video_ocr.py
```

---

# 🚗 Vehicle Tracking

Run tracking pipeline:

```bash
python src/tracking/vehicle_tracking.py
```

---

# 📈 Traffic Analytics

Run analytics system:

```bash
python src/analytics/traffic_analytics.py
```

---

# 💻 Installation

# 1️⃣ Clone Repository

```bash
git clone https://github.com/shubhampandey97/number-plate-vehicle-detection.git

cd number-plate-vehicle-detection
```

---

# 2️⃣ Create Environment

```bash
conda create -n npvd_env python=3.11

conda activate npvd_env
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Train Model

```bash
python training/train.py
```

---

# 5️⃣ Run Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

# 📦 Requirements

Main dependencies:

```text
ultralytics
opencv-python
easyocr
streamlit
numpy
pandas
torch
torchvision
lap
```

---

# 📷 Output Examples

The system generates:

✅ Vehicle detection outputs  
✅ OCR-annotated outputs  
✅ Tracking outputs  
✅ Traffic analytics videos  
✅ CSV plate logs  
✅ Real-time monitoring dashboard  

---

# 📌 Key Learnings

Through this project, the following concepts were implemented deeply:

- Transfer Learning
- YOLOv8 Architecture
- Object Detection Pipelines
- OCR Integration
- ByteTrack Tracking
- Real-time Video Processing
- Traffic Analytics
- Streamlit Dashboard Development
- GPU-Accelerated Deep Learning
- End-to-End AI Pipeline Development

---

# 📉 Current Challenges

Some OCR challenges still exist for:
- Motion blur
- Tiny number plates
- Extreme viewing angles
- Low-light conditions

---

# 🔮 Future Improvements

Potential future enhancements:
- Speed estimation
- Lane detection
- Helmet detection
- Red-light violation detection
- Parking analytics
- Cloud deployment
- Database integration
- REST API support
- Super-resolution for OCR
- Advanced OCR models

---

# 💼 Portfolio Value

This project demonstrates:
- Deep Learning Engineering
- Computer Vision Skills
- OCR System Development
- Real-time AI Systems
- Object Tracking
- AI Dashboard Development
- Production-style ML Pipeline Structuring

---

# 👨‍💻 Author

## Shubham Pandey

Data Science & AI Developer  
Computer Vision & Deep Learning Enthusiast

---

# ⭐ If You Like This Project

Give this repository a ⭐ on GitHub and feel free to contribute or provide suggestions.