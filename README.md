# SonoYOLO + MedSAM2: A Pipeline for Automatic Detection and Segmentation of Regions of Interest in Cardiac Ultrasound Videos

Abstract:
Precise evaluation of cardiac function is essential for early diagnosis, risk assessment, and treatment of cardiovascular diseases. Cardiac ultrasound videos are widely used for this purpose, but cardiologists need to manually delineate the regions of interest in the videos, which is a time-consuming and error-prone process. While many automatic video segmentation methods exist, modern deep learning-based methods require dense annotations for training, which is labor-intensive. Recent zero-shot segmentation models specialized for medical data such as MedSAM2 do not require such dense annotations, but they still need manual prompts (e.g., bounding boxes), which hinders automatic deployment. To address these challenges, we propose a fully automated pipeline for detecting and segmenting regions of interest in cardiac ultrasound videos. The proposed pipeline (SonoYOLO + MedSAM2) consists of a detection model (SonoYOLO) obtained by finetuning YOLOv11-nano to localize the LV in an image, and a segmentation model (MedSAM2) that receives a bounding box as a prompt and segments the LV. The resulting image mask is used as a prompt to segment the next image and so on, thus achieving temporal consistency across the video. Experiments on EchoNet-Dynamic and EchoNet-Pediatric datasets show strong performance and generalizability of our pipeline without requiring dense annotations or manual prompts. 

Download the Stanford EchoNet-Dynamic Dataset from: https://echonet.github.io/dynamic/index.html#dataset
Have two separate conda environments for SonoYOLO detection and MedSAM2 inference. I have given the requirements.txt files for both environments, but you can follow the directions below.

In the SonoYOLO environment, you can look how to set YOLOv11n from here: https://github.com/ultralytics/ultralytics, or just pip install ultralytics. My SonoYOLO.ipynb file has everything else needed for YOLO.
Once SonoYOLO environment is set up, run SonoYOLO.ipynb, and then set up MedSAM2 environment.

MedSAM2 gives their environmental setup instructions (https://github.com/bowang-lab/MedSAM2): 
Installation
Create a virtual environment: conda create -n medsam2 python=3.12 -y and conda activate medsam2
Install PyTorch: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 (Linux CUDA 12.4)
Download code git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2 and run pip install -e ".[dev]"
Download checkpoints: bash download.sh
Optional: Please install the following dependencies for gradio
sudo apt-get update
sudo apt-get install ffmpeg
pip install gradio==3.38.0
pip install numpy==1.26.3 
pip install ffmpeg-python 
pip install moviepy

Additionally, depending on how you run the code you may need additional packages which is why I recommend checking outt the requirements.txt. Once MedSAM2 is set up you can run SonoYolo + MedSAM2 Inference.ipynb. This will get you the results for our method.
