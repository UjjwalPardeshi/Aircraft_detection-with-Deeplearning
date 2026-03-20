
# Aircraft & Drone Detection with Deep Learning

Real-time aerial object detection system for identifying **birds** and **drones** using YOLOv9 and YOLOv10 architectures. Trained and evaluated on the Det-Fly dataset (4,689 training images, 1,562 validation images) with comparative analysis across model variants.

## Highlights

- **Best mAP@50: 93.8%** (YOLOv9c fine-tuned, 30 epochs)
- **Fastest inference: 3.3ms/image** (YOLOv10n)
- 2-class detection: Bird & Drone
- Trained on Tesla T4 GPU using Roboflow + Ultralytics pipeline

## Results Comparison

| Model | Params | Epochs | mAP@50 | mAP@50-95 | Precision | Recall | Inference |
|-------|--------|--------|--------|-----------|-----------|--------|-----------|
| YOLOv9c | 25.5M | 25 | 0.927 | 0.540 | 0.920 | 0.887 | 14.8ms |
| YOLOv9c (fine-tuned) | 25.5M | 30 | **0.938** | **0.575** | **0.936** | **0.899** | 14.8ms |
| YOLOv10n | 2.7M | 30 | 0.810 | 0.427 | — | — | **3.3ms** |

### Per-Class Performance (Best Model — YOLOv9c Fine-tuned)

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| Bird | 0.960 | 0.579 |
| Drone | 0.917 | 0.570 |

## Project Structure

```
├── Dataset_Drone_bird_aircraft/
│   └── yolo_v9.ipynb              # YOLOv9c baseline training (25 epochs)
├── Finetuned on det_fly/
│   ├── finetuning_with_detfly.ipynb   # YOLOv9c fine-tuning (30 epochs) ← best results
│   ├── Results/
│   └── Weights/
├── Yolo_V10/
│   ├── yolov10_det_fly.ipynb      # YOLOv10n lightweight model
│   ├── Results/
│   └── Weights/
└── README.md
```

## Dataset

**Det-Fly v6** sourced from [Roboflow](https://roboflow.com)

| Split | Images | Instances |
|-------|--------|-----------|
| Train | 4,689 | ~6,251 |
| Val | 1,562 | ~3,142 |
| **Classes** | **Bird, Drone** | |

## Tech Stack

- **Framework:** Ultralytics (YOLOv9, YOLOv10)
- **Deep Learning:** PyTorch 2.3 + CUDA 12.1
- **Dataset Management:** Roboflow API
- **Augmentation:** Albumentations (blur, median blur, grayscale, CLAHE)
- **Optimizer:** AdamW (lr=0.00167, batch=16, img=640×640)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/UjjwalPardeshi/Aircraft_detection-with-Deeplearning.git
cd Aircraft_detection-with-Deeplearning

# Install dependencies
pip install ultralytics roboflow torch torchvision opencv-python matplotlib

# Run any notebook
jupyter notebook "Finetuned on det_fly/finetuning_with_detfly.ipynb"
```

## Use Cases

- Surveillance & airspace monitoring
- Aviation safety — detecting unauthorized drones near airports
- Wildlife monitoring — bird tracking and counting
- Defense — aerial threat classification

## License

This project is open source. Feel free to use and modify for research and educational purposes.
