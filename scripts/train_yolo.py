"""
YOLO Transfer Learning & Fine-Tuning Script for Airshow & UAV Footage (Week 2+).

Run this script at home (with GPU / PyTorch) to fine-tune YOLO weights on custom
labeled footage or domain datasets (VisDrone / DOTA / Aerial Maritime).

Usage:
    py -3.10 scripts/train_yolo.py --data config/uav_dataset.yaml --weights yolo11s.pt --epochs 50 --imgsz 1024

Outputs:
    - Saved weights in models/yolo/yolo11_uav_custom.pt
    - Copy to models/yolo/ and update hashes.sha256 before transferring to facility.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_uav_yolo(
    data_yaml: str,
    base_weights: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 1024,
    batch_size: int = 16,
    output_name: str = "yolo11_uav_custom.pt",
    freeze_backbone: int = 10,
):
    print(f"Loading base weights: {base_weights}")
    model = YOLO(base_weights)

    print(f"Starting Transfer Learning on dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Image Size: {imgsz}, Batch Size: {batch_size}, Freeze layers: {freeze_backbone}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        freeze=freeze_backbone,
        project="runs/detect",
        name="uav_finetune",
        exist_ok=True,
    )

    best_weights = Path("runs/detect/uav_finetune/weights/best.pt")
    target_path = Path("models/yolo") / output_name

    if best_weights.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(best_weights, target_path)
        print(f"Transfer learning complete! Saved fine-tuned weights to: {target_path}")
        print(f"Remember to compute SHA256 and update hashes.sha256 before facility transfer!")
    else:
        print("Training finished but best.pt was not found. Check runs/detect output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for UAV & Aerial footage")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset data.yaml file")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="Base model weights (e.g. yolo11s.pt, yolov8x-visdrone.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference / training image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="yolo11_uav_custom.pt", help="Output model filename")

    args = parser.parse_args()
    train_uav_yolo(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_name=args.output,
    )
