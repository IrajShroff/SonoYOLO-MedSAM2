# SonoYOLO + MedSAM2: A Fully Automated Pipeline for Left Ventricle Detection and Segmentation in Cardiac Ultrasound Videos

## Abstract
Precise evaluation of cardiac function is essential for early diagnosis, risk assessment, and treatment of cardiovascular diseases. Cardiac ultrasound videos are widely used for this purpose, but cardiologists need to manually delineate the regions of interest in the videos, which is a time-consuming and error-prone process. While many automatic video segmentation methods exist, modern deep learning-based methods require dense annotations for training, which is labor-intensive. Recent zero-shot segmentation models specialized for medical data such as MedSAM2 do not require such dense annotations, but they still need manual prompts (e.g., bounding boxes), which hinders automatic deployment. To address these challenges, we propose a fully automated pipeline for detecting and segmenting regions of interest in cardiac ultrasound videos. The proposed pipeline (SonoYOLO + MedSAM2) consists of a detection model (SonoYOLO) obtained by finetuning YOLOv11-nano to localize the LV in an image, and a segmentation model (MedSAM2) that receives a bounding box as a prompt and segments the LV. The resulting image mask is used as a prompt to segment the next image and so on, thus achieving temporal consistency across the video. Experiments on EchoNet-Dynamic and EchoNet-Pediatric datasets show strong performance and generalizability of our pipeline without requiring dense annotations or manual prompts.

---

## Dataset
Download the Stanford **EchoNet-Dynamic** dataset:  
https://echonet.github.io/dynamic/index.html#dataset

(Optional) EchoNet-Pediatric can also be used.

---

## Repository Structure
```
.
├── SonoYOLO.ipynb
├── SonoYOLO + MedSAM2 Inference.ipynb
├── sonoyolo_requirements.txt
├── medsam2_requirements.txt
└── README.md
```

---

## Environment Setup

This pipeline uses **two separate conda environments**:

| Environment | Purpose |
|------------|---------|
| `sonoyolo-env` | LV bounding box detection |
| `medsam2` | Segmentation + temporal propagation |

---

### 1. Create & Configure SonoYOLO Environment

```bash
conda create -n sonoyolo-env python=3.10 -y
conda activate sonoyolo-env
pip install ultralytics
```

Run training / inference:

```bash
jupyter notebook SonoYOLO.ipynb
```

---

### 2. Create & Configure MedSAM2 Environment

```bash
conda create -n medsam2 python=3.12 -y
conda activate medsam2
```

Install PyTorch (CUDA 12.4 example):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Download MedSAM2 and install:

```bash
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"
bash download.sh
```

**Optional** (video visualization support):

```bash
sudo apt-get update && sudo apt-get install ffmpeg
pip install gradio==3.38.0 numpy==1.26.3 ffmpeg-python moviepy
```

---

## Running the Pipeline

### Step 1: LV Detection with SonoYOLO

```bash
conda activate sonoyolo-env
jupyter notebook SonoYOLO.ipynb
```

This generates LV bounding box predictions.

### Step 2: Segmentation with MedSAM2

```bash
conda activate medsam2
jupyter notebook "SonoYOLO + MedSAM2 Inference.ipynb"
```

This:
- Uses YOLO bounding boxes as prompts
- Segments LV with MedSAM2
- Propagates segmentation through video frames

---

## Output
You will obtain:
- Per-frame segmentation masks
- (Optional) Segmented video reconstruction
- (Optional) Metrics if configured

---

## Citation
If used in academic work, please cite:
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- MedSAM2: https://github.com/bowang-lab/MedSAM2

---
