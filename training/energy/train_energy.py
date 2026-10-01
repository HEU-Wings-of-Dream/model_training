"""Train the latest energy-mechanism pose network."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the energy-mechanism pose model")
    parser.add_argument("--cfg", default=str(ROOT / "mobilenet_energy_pose.yaml"))
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(480, 640))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="energy_pose")
    args = parser.parse_args()

    model = YOLO(args.cfg, task="pose")
    if args.pretrained:
        model.load(args.pretrained)

    model.train(data=args.data, epochs=args.epochs, imgsz=tuple(args.imgsz),
                batch=args.batch, device=args.device, workers=args.workers,
                amp=True, task="pose", lr0=0.008, degrees=30, patience=80,
                perspective=0.0002, save_period=20, scale=0.05,
                close_mosaic=110, hsv_v=0.5, erasing=0.2, fliplr=0,
                box=7.5, cls=0.2, dfl=3, pose=12, kobj=2, cache=False,
                project="runs", name=args.name)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
