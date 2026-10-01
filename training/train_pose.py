"""Train a YOLO pose model.

Example:
    python training/train_pose.py --data path/to/data.yaml --model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import multiprocessing

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO pose model")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="pose_train")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=0.008,
        patience=40,
        perspective=0.0002,
        save_period=20,
        scale=0.0005,
        close_mosaic=90,
        hsv_h=0.005,
        hsv_s=0.005,
        hsv_v=0.7,
        project="runs",
        name=args.name,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
