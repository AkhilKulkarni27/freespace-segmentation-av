# Free Space Detection — Level 4 Autonomous Vehicle

Real-time road segmentation for Level 4 autonomous vehicles using DeepLabV3+ with EfficientNet-B4 backbone. The model identifies driveable free space — areas where the vehicle can physically move — regardless of lane markings, in complex urban settings.

---

## Project Overview

Level 4 autonomous vehicles must identify **Free Space**: areas where the car can physically move, regardless of whether lane markings exist. This project focuses on segmenting the road vs. everything else (curbs, construction barriers, sidewalks) in complex urban settings.

### Key Results

| Metric | Score |
|--------|-------|
| Validation IoU | **92.2%** |
| Validation F1 | **95.9%** |
| Inference Speed | **50.8 FPS** |
| Inference Time | 19.7 ms/frame |
| Pretrained Weights | None (trained from scratch) |

---

## Model Architecture

- **Architecture**: DeepLabV3+ (Encoder-Decoder with Atrous Convolutions)
- **Backbone**: EfficientNet-B4 (real-time capable)
- **Parameters**: 18.6M
- **Input size**: 512 x 512
- **Output**: Binary segmentation mask (free space vs. not free space)

### Loss Function
Combined Dice Loss + Weighted Binary Cross Entropy:
- Dice Loss handles class imbalance
- Weighted BCE gives 2x weight to road pixels and 4x weight to boundary pixels (curbs, barriers)

### Training Improvements
- Deep Supervision — auxiliary losses at intermediate layers
- Class Weights — higher importance to boundary pixels
- Hard Negative Mining — increased sampling of failure cases from epoch 5
- Mixup Augmentation — blends training images for robustness from epoch 10
- Heavy Weather Augmentation — fog, rain, shadow, sun flare, motion blur

---

## Dataset

### nuScenes (mini)
- 1,212 image/mask pairs
- All 6 camera angles (front, back, left, right)
- Urban driving in Boston and Singapore
- https://www.nuscenes.org/nuscenes

### Cityscapes
- 3,475 image/mask pairs
- Official pixel-level annotations
- 50 cities across Europe
- https://www.cityscapes-dataset.com

### Combined Split
| Split | Images |
|-------|--------|
| Train | 3,280 |
| Val | 703 |
| Test | 704 |
| Total | 4,687 |

---

## Setup & Installation

### Requirements
- Python 3.11
- NVIDIA GPU with CUDA 12.8+
- 12GB+ disk space

### Installation

```bash
git clone https://github.com/AkhilKulkarni27/freespace-segmentation-av
cd freespace-segmentation-av
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## How to Run

### Train the model
```bash
python run_all_v3.py
```

### Run inference on an image
```bash
python inference.py --checkpoint checkpoints_v3/best_model.pth --image your_image.jpg
```

### Run inference on a video
```bash
python inference.py --checkpoint checkpoints_v3/best_model.pth --video your_video.mp4 --output result.mp4
```

### Benchmark FPS
```bash
python benchmark_fps.py
```

### Generate Grad-CAM visualization
```python
from run_all_v3 import visualize_gradcam
visualize_gradcam('your_image.jpg', 'checkpoints_v3/best_model.pth')
```

### Interactive Colab Demo
Open `FreespaceDemo_v2.ipynb` in Google Colab, upload `best_model.pth` to Google Drive, enable T4 GPU and run all cells.

---

## Example Results

### Training Progress

| Epoch | Train IoU | Val IoU | Val F1 |
|-------|-----------|---------|--------|
| 10 | 0.623 | 0.741 | 0.851 |
| 25 | 0.801 | 0.878 | 0.935 |
| 50 | 0.865 | 0.906 | 0.951 |
| 75 | 0.887 | 0.918 | 0.958 |
| 100 | 0.896 | 0.922 | 0.959 |

### Performance vs Baselines

| Model | Backbone | Pretrained | IoU | FPS |
|-------|----------|------------|-----|-----|
| Baseline | ResNet50 | Yes | 90.9% | 51 |
| Ours | EfficientNet-B4 | No | 92.2% | 50.8 |
| Industry typical | ResNet101 | Yes | ~88-91% | ~30 |

---

## Project Structure

```
freespace-segmentation-av/
├── run_all_v3.py            # Main training pipeline
├── inference.py             # Image and video inference
├── dataset.py               # Dataset loader and augmentation
├── model.py                 # Model definition and metrics
├── benchmark_fps.py         # FPS benchmark script
├── make_demo_video.py       # Demo video generation
├── FreespaceDemo_v2.ipynb   # Google Colab interactive demo
├── requirements.txt         # Dependencies
└── README.md
```

---

## Problem Statement

> Level 4 vehicles must identify Free Space — areas where the car can physically move — regardless of whether lane markings exist. This track focuses on segmenting the road vs. everything else (curbs, construction barriers, sidewalks) in complex urban settings.

### How we address it
- No lane marking dependency — model uses surface texture and context
- Urban complexity — trained on nuScenes and Cityscapes across 50+ cities
- All obstacles excluded — curbs, barriers, sidewalks, parked cars, pedestrians
- Real-time capable — 50.8 FPS, deployable on edge hardware
- No pretrained weights — trained entirely from scratch as required
