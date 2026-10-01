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

The latest specialized networks are organized separately:

```text
training/energy/   energy-mechanism network
training/armor/    armor corner network
```

Each directory contains its training entry point, network YAML, and dataset
configuration template. Both specialized YAML files use custom Ultralytics
layers (`EnergyPoseDetect`, `ArmorPoseDetect`, and `MobileNetV3IR`) from the
local training environment; the corresponding custom Ultralytics source or
package must be installed before training.

Examples:

```bash
python training/energy/train_energy.py --data training/energy/data.yaml
python training/armor/train_armor_corners.py --data training/armor/data.yaml
```

Useful options include `--epochs`, `--batch`, `--device`, `--workers`, and
`--name`. Training outputs are written under `runs/`.

## Local-only files

Keep large or machine-specific files such as datasets, `.pt`/`.onnx`
weights, videos, caches, and experiment outputs outside version control.
