# SonoYOLO + MedSAM2: A Fully Automated Pipeline for Left Ventricle Detection and Segmentation in Cardiac Ultrasound Videos

## Abstract
Precise evaluation of cardiac function is essential for early diagnosis, risk assessment, and treatment of cardiovascular diseases. Cardiac ultrasound videos are widely used for this purpose, but cardiologists need to manually delineate the regions of interest in the videos, which is a time-consuming and error-prone process. While many automatic video segmentation methods exist, modern deep learning-based methods require dense annotations for training, which is labor-intensive. Recent zero-shot segmentation models specialized for medical data such as MedSAM2 do not require such dense annotations, but they still need manual prompts (e.g., bounding boxes), which hinders automatic deployment. To address these challenges, we propose a fully automated pipeline for detecting and segmenting regions of interest in cardiac ultrasound videos. The proposed pipeline (SonoYOLO + MedSAM2) consists of a detection model (SonoYOLO) obtained by finetuning YOLOv11-nano to localize the LV in an image, and a segmentation model (MedSAM2) that receives a bounding box as a prompt and segments the LV. The resulting image mask is used as a prompt to segment the next image and so on, thus achieving temporal consistency across the video. Experiments on EchoNet-Dynamic and EchoNet-Pediatric datasets show strong performance and generalizability of our pipeline without requiring dense annotations or manual prompts.

---

## Dataset
Download the **Stanford EchoNet-Dynamic dataset**:  
https://echonet.github.io/dynamic/index.html#dataset

---

## Repository Structure
```
.
├── SonoYOLO_Training.py
├── SonoYOLO_Testing.py
├── SonoYolo + MedSAM2 forward propagation.ipynb
├── SonoYolo + MedSAM2 backward propagation.ipynb
├── SonoYolo + MedSAM2 frame-wise.ipynb
├── sonoyolo_requirements.txt
├── medsam2_requirements.txt
├── echonet_yolo.yaml
└── README.md
```

---

## Environment Setup

This pipeline uses **two separate conda environments**:

| Environment | Purpose |
|------------|---------|
| `sonoyolo-env` | Left Ventricle bounding box detection using SonoYOLO |
| `medsam2` | LV segmentation + temporal propagation using MedSAM2 |

---

# 1. SonoYOLO Environment (Detection)

### Create & Configure SonoYOLO Environment

```bash
conda create -n sonoyolo-env python=3.10 -y
conda activate sonoyolo-env
pip install ultralytics
```

### Training (YOLOv11-nano finetuning)
```bash
python SonoYOLO_Training.py
```

### Testing (SonoYOLO LV predicted bounding boxes)
```bash
python SonoYOLO_Testing.py
```

This produces YOLO-format bounding box files that will be used as prompts for MedSAM2.

---

# 2. MedSAM2 Environment (Segmentation)

### Create & Configure MedSAM2 Environment

```bash
conda create -n medsam2 python=3.12 -y
condda activate medsam2
```

### Install PyTorch (CUDA 12.4 example)
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### Download & Install MedSAM2
```bash
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"
bash download.sh
```

### Optional (video visualization)
```bash
sudo apt-get update && sudo apt-get install ffmpeg
pip install gradio==3.38.0 numpy==1.26.3 ffmpeg-python moviepy
```

---

# Running the Pipeline

After generating bounding boxes with SonoYOLO, you can run **three different segmentation methods**, each corresponding to a method from the paper.

## Method 1 — Forward Propagation 
Refer to paper to understand method.
```bash
conda activate medsam2
jupyter notebook "SonoYolo + MedSAM2 forward propagation.ipynb"
```

## Method 2 — Backward Propagation
Refer to paper to understand method.

```bash
conda activate medsam2
jupyter notebook "SonoYolo + MedSAM2 backward propagation.ipynb"
```

## Method 3 — Frame-wise Zero-Shot Segmentation
Runs MedSAM2 independently on each frame using only SonoYOLO bounding boxes as prompts.

```bash
conda activate medsam2
jupyter notebook "SonoYolo + MedSAM2 frame-wise.ipynb"
```


---

## Output

You will obtain:

- **Per-frame Left Ventricle segmentation masks**
- **CSV DSC logs** (if enabled)

---

## Citation
If this repository is used in academic work, please cite:

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics  
- MedSAM2: https://github.com/bowang-lab/MedSAM2  
