#!/usr/bin/env python
# coding: utf-8

# # SonoYolo: Finetuning YOLOv11n

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

#Data Preparation for training and testing
import os
import cv2
import pandas as pd
from tqdm import tqdm

#  CONFIGURATION 
video_dir = "/EchoNet-Dynamic/Videos"
output_root = "echonet_yolo"
original_size = 112         # Original size of frames (EchoNet-Dynamic Dataset
padded_size = 128           # Padded file size for YOLOv11n
pad = (padded_size - original_size) // 2  # 8 pixels on each side

vol_csv = "/EchoNet-Dynamic/VolumeTracings.csv" #Volume Tracings File in Stanford Echonet-Dynamic Dataset File
filelist_csv = "/EchoNet-Dynamic/FileList.csv" #We are using the splits that the Stanford EchoNet-Dynamic Model used for training, testing and validation

#  Create output folders 
os.makedirs(output_root, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_root, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_root, f"labels/{split}"), exist_ok=True)

#  Load annotations (VolumeTracings.csv) 
df = pd.read_csv(vol_csv)
if "FileName" in df.columns:
    tracings_file_col = "FileName"
elif "Filename" in df.columns:
    tracings_file_col = "Filename"
else:
    raise ValueError("VolumeTracings.csv must contain 'FileName' or 'Filename' column.")

df[tracings_file_col] = df[tracings_file_col].astype(str).str.strip()

#  Load EchoNet-style split list (FileList.csv) 
fl = pd.read_csv(filelist_csv)


if "FileName" in fl.columns:
    filelist_file_col = "FileName"
elif "Filename" in fl.columns:
    filelist_file_col = "Filename"
else:

    filelist_file_col = fl.columns[0]

if "Split" in fl.columns:
    split_col = "Split"
elif "split" in fl.columns:
    split_col = "split"
else:
    raise ValueError("FileList.csv must contain 'Split' (or 'split') column with values TRAIN/VAL/TEST.")

# Normalize filenames (as given, these are basenames without extension)
fl[filelist_file_col] = fl[filelist_file_col].astype(str).str.strip()

# Normalize split labels from TRAIN/VAL/TEST -> train/val/test
fl[split_col] = (
    fl[split_col]
    .astype(str).str.strip().str.upper()
    .map({"TRAIN": "train", "VAL": "val", "TEST": "test"})
)


fl["_with_ext"] = fl[filelist_file_col] + ".avi"

# dataset splits from FileList.csv
splits = {
    'train': sorted(fl.loc[fl[split_col] == 'train', "_with_ext"].unique().tolist()),
    'val':   sorted(fl.loc[fl[split_col] == 'val',   "_with_ext"].unique().tolist()),
    'test':  sorted(fl.loc[fl[split_col] == 'test',  "_with_ext"].unique().tolist()),
}

print({k: len(v) for k, v in splits.items()})

grouped = {k: g for k, g in df.groupby(tracings_file_col)}

#  MAIN LOOP 
for split, video_list in splits.items():
    img_dir = os.path.join(output_root, f"images/{split}")
    lbl_dir = os.path.join(output_root, f"labels/{split}")

    for video_name in tqdm(video_list, desc=f"Processing {split}"):
        # Skip if no tracings (paranoia; we already filtered)
        if video_name not in grouped:
            continue

        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_name}")
            continue

        # Get annotated frames for this video
        frames_df = grouped[video_name]
        unique_frames = frames_df['Frame'].dropna().astype(int).unique()
        if len(unique_frames) < 2:
            cap.release()
            continue

        # Choose first 2 annotated frames (ED/ES typically present)
        selected_frames = sorted(unique_frames)[:2]

        for frame_num in selected_frames:
            frame_rows = frames_df[frames_df['Frame'].astype(int) == frame_num]
            if frame_rows.empty:
                continue

            # Get bbox corners from tracings
            x_vals = pd.concat([frame_rows['X1'], frame_rows['X2']])
            y_vals = pd.concat([frame_rows['Y1'], frame_rows['Y2']])

            xmin, xmax = int(x_vals.min()), int(x_vals.max())
            ymin, ymax = int(y_vals.min()), int(y_vals.max())

            # Read frame (EchoNet videos are already 112x112)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_num} in {video_name}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # === Pad to 128×128 ===
            frame_padded = cv2.copyMakeBorder(
                frame_rgb, pad, pad, pad, pad,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            # === Adjust bounding box (shift by pad; normalize by 128) ===
            x_center = ((xmin + xmax) / 2 + pad) / padded_size
            y_center = ((ymin + ymax) / 2 + pad) / padded_size
            width = (xmax - xmin) / padded_size
            height = (ymax - ymin) / padded_size

            # === Save image and label ===
            base_name = f"{video_name.replace('.avi', '')}_frame{frame_num}"
            image_path = os.path.join(img_dir, f"{base_name}.jpg")
            label_path = os.path.join(lbl_dir, f"{base_name}.txt")

            cv2.imwrite(image_path, frame_padded)
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        cap.release()

print("Done.")

#Training Section

from ultralytics import YOLO
from ultralytics import YOLO
import os

# Edit based on your needs
DATA_YAML = "/echonet_yolo/echonet_yolo.yaml" #this yaml is given in the repository
TRAIN_EPOCHS = 40
IMG_SIZE = 128
BATCH_SIZE = 16
TRAIN_WEIGHTS = "yolo11n.pt"  # pretrained weights to start from


# Training

print("Loading pretrained YOLOv11n model")
model = YOLO(TRAIN_WEIGHTS)

print(f"Training model on dataset: {DATA_YAML}")
model.train(
    data=DATA_YAML,
    epochs=TRAIN_EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    pretrained=True  # use pretrained weights and adapt head
)