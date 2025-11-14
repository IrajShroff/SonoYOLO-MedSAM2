# SonoYOLO_Testing.py

# This script runs the trained SonoYOLO detector on the EchoNet-Dynamic *test* subset.
# It generates bounding-box predictions for every test-frame image and saves:
#   - YOLO-format text files with bounding box coordinates (if save_txt=True)
# These predicted coordinates will later be used in the full SonoYOLO + MedSAM2 segmentation pipeline and in quantitative evaluation of the detection stage.


from ultralytics import YOLO

def main():
    # Path to the trained model weights.
    # After training, YOLO stores 'best.pt' in runs/detect/train/weights/.
    # Update this path if your training directory is different.
    best_weights_path = "/runs/detect/train/weights/best.pt"

    model = YOLO(best_weights_path)

    results = model.predict(
        source="/echonet_yolo/images/test",  # Directory containing test-set images
        imgsz=128,                           # Must match the training image size
        conf=0.65,                           # Confidence threshold
        save=True,                           # Save annotated images to runs/detect/exp/
        save_txt=True                        # Save YOLO-format bbox text files
    )

if __name__ == "__main__":
    main()
