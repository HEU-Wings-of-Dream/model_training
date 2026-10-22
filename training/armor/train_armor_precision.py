"""Train the accuracy-first, single-class armor corner pose model."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the high-accuracy armor corner pose model")
    parser.add_argument("--cfg", type=Path, default=ROOT / "armor11_corners_precision_pose.yaml")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics pose dataset YAML")
    parser.add_argument("--pretrained", default="yolov8n-pose.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="armor11_corners_precision")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.cfg.is_file():
        raise FileNotFoundError(f"Model YAML not found: {args.cfg}")

    import torch
    from ultralytics import YOLO
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import head

    # The original architecture names the project's pose head ArmorPoseDetect.
    # Register the compatible Ultralytics Pose head for standalone training.
    if not hasattr(head, "ArmorPoseDetect"):
        head.ArmorPoseDetect = head.Pose
    if not hasattr(tasks, "ArmorPoseDetect"):
        tasks.ArmorPoseDetect = head.ArmorPoseDetect

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA-enabled PyTorch environment.")
    if str(args.device).isdigit():
        torch.cuda.set_device(int(args.device))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = YOLO(str(args.cfg), task="pose")
    if args.pretrained:
        model.load(args.pretrained)
    model.train(
        data=str(args.data), epochs=args.epochs, batch=args.batch,
        imgsz=tuple(args.imgsz), device=args.device, workers=args.workers,
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=5, patience=100, cos_lr=True, degrees=10.0,
        translate=0.05, scale=0.15, shear=2.0, perspective=0.0001,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.3, fliplr=0.0, mosaic=0.5,
        close_mosaic=30, box=8.0, cls=0.01, dfl=3.0, pose=30.0,
        kobj=3.0, amp=True, cache=args.cache, plots=False,
        deterministic=False, save_period=20, project="runs", name=args.name,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
