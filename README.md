# Model training

This repository contains the model-training code extracted from the local
workspace. Dataset files, model weights, videos, generated `runs/` output,
and virtual environments are intentionally kept out of Git.

## Environment

```bash
python -m pip install ultralytics
```

Copy `training/data.yaml.example` to `training/data.yaml`, update the dataset
paths, and then run:

```bash
python training/train_pose.py --data training/data.yaml --model yolo11n-pose.pt
```

Useful options include `--epochs`, `--batch`, `--device`, `--workers`, and
`--name`. Training outputs are written under `runs/`.

## Local-only files

Keep large or machine-specific files such as datasets, `.pt`/`.onnx`
weights, videos, caches, and experiment outputs outside version control.
