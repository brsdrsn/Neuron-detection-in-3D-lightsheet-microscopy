"""
Pipeline stage: 4 — YOLO training
=================================
Train (or fine-tune) a YOLOv8 detector on the dataset produced by train_yolo.py.

Inputs:  ~/Documents/neurons/yolo_tagged_tiles_v3/data.yaml  (--data to override)
Outputs: runs/detect/<name>/weights/best.pt  (default name: train_yolo_tagged_tiles_v3)

Run:
  python train_yolo_weights.py
  python train_yolo_weights.py --epochs 200 --batch 8 --device 0

Suggested rename: train_yolo.py  (after renaming current train_yolo.py → build_yolo_dataset.py)
"""

from pathlib import Path
import argparse
from ultralytics import YOLO


def parse_args():
    """CLI for YOLO training; defaults match the v3 tagged-tiles dataset."""
    neurons_root = Path.home() / "Documents" / "neurons"
    default_data = neurons_root / "yolo_tagged_tiles_v3" / "data.yaml"
    default_project = Path(__file__).resolve().parent / "runs" / "detect"

    parser = argparse.ArgumentParser(description="Train YOLO on yolo_tagged_tiles_v3 (or pass --data).")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or weights path.")
    parser.add_argument("--data", default=str(default_data), help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (Ultralytics). Use 0 to disable early stopping.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument(
        "--project",
        default=str(default_project),
        help="Output project directory for runs.",
    )
    parser.add_argument(
        "--name",
        default="train_yolo_tagged_tiles_v3",
        help="Run name inside project directory.",
    )
    parser.add_argument(
        "--device",
        default="0",  # use first CUDA GPU
        help="Training device, e.g. 0, 0,1, or cpu.",
    )
    parser.add_argument("--workers", type=int, default=0)   # must be 0 on Windows (DataLoader)
    parser.add_argument("--batch", type=int, default=8)     # lower if GPU OOM
    parser.add_argument("--close_mosaic", type=int, default=0)  # optional workaround
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print(f"Training with data: {data_path}")
    print(f"Model: {args.model}")
    print(
        f"Epochs: {args.epochs}, imgsz: {args.imgsz}, device: {args.device}, patience: {args.patience}"
    )

    # Fine-tune from COCO-pretrained yolov8n (or pass custom --model weights)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        cache=False,
    )


if __name__ == "__main__":
    main()
