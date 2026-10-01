"""Train the latest armor corner pose network."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the armor corner pose model")
    parser.add_argument("--cfg", default=str(ROOT / "armor11_corners_lite_pose.yaml"))
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--pretrained", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(480, 640))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="armor11_corners")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.cfg, task="pose")
    if args.pretrained:
        model.load(args.pretrained)

    model.train(data=args.data, epochs=args.epochs, imgsz=tuple(args.imgsz),
                batch=args.batch, device=args.device, workers=args.workers,
                lr0=0.006, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                warmup_epochs=3, patience=90, degrees=30, translate=0.08,
                scale=0.08, perspective=0.0002, hsv_h=0.015, hsv_s=0.7,
                hsv_v=0.45, erasing=0.15, fliplr=0, close_mosaic=120,
                box=7.5, cls=0.01, dfl=2.5, pose=18.0, kobj=2.5,
                amp=True, cache=args.cache, save_period=20, project="runs",
                name=args.name)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
