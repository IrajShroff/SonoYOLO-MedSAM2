# SonoYOLO + MedSAM2: A Pipeline for Automatic Detection and Segmentation of Regions of Interest in Cardiac Ultrasound Videos

## Abstract
Precise evaluation of cardiac function is essential for early diagnosis, risk assessment, and treatment of cardiovascular diseases. Cardiac ultrasound videos are widely used for this purpose, but cardiologists need to manually delineate the regions of interest in the videos, which is a time-consuming and error-prone process. While many automatic video segmentation methods exist, modern deep learning-based methods require dense annotations for training, which is labor-intensive. Recent zero-shot segmentation models specialized for medical data such as MedSAM2 do not require such dense annotations, but they still need manual prompts (e.g., bounding boxes), which hinders automatic deployment. To address these challenges, we propose a fully automated pipeline for detecting and segmenting regions of interest in cardiac ultrasound videos. The proposed pipeline (SonoYOLO + MedSAM2) consists of a detection model (SonoYOLO) obtained by finetuning YOLOv11-nano to localize the LV in an image, and a segmentation model (MedSAM2) that receives a bounding box as a prompt and segments the LV. The resulting image mask is used as a prompt to segment the next image and so on, thus achieving temporal consistency across the video. Experiments on EchoNet-Dynamic and EchoNet-Pediatric datasets show strong performance and generalizability of our pipeline without requiring dense annotations or manual prompts.

## Dataset
Download the Stanford EchoNet-Dynamic Dataset:  
https://echonet.github.io/dynamic/index.html#dataset

## Environment Setup Overview
This pipeline uses two separate conda environments:
- SonoYOLO (LV detection)
- MedSAM2 (LV segmentation)

## 1. Setup SonoYOLO Environment
conda create -n sonoyolo-env python=3.10 -y
conda activate sonoyolo-env
pip install ultralytics

Run `SonoYOLO.ipynb` to perform training and inference.

## 2. Setup MedSAM2 Environment
conda create -n medsam2 python=3.12 -y
conda activate medsam2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"
bash download.sh

(Optional visualization dependencies):
sudo apt-get update && sudo apt-get install ffmpeg
pip install gradio==3.38.0 numpy==1.26.3 ffmpeg-python moviepy

## Running the Pipeline
1. Run `SonoYOLO.ipynb` to generate bounding box detections.
2. Activate medsam2 environment and run `SonoYOLO + MedSAM2 Inference.ipynb`.

## Citation
If used, please cite YOLO and MedSAM2 original works. 