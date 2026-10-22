# Latest training architectures

This repository contains only the latest training architectures for the
energy mechanism and armor auto-aim. Datasets, weights, videos, generated
`runs/` output, and virtual environments are intentionally excluded.

## Dependencies

```bash
python -m pip install ultralytics
```

Both architectures require the local Ultralytics extensions that provide
`EnergyPoseDetect`, `ArmorPoseDetect`, and `MobileNetV3IR`.

## Energy mechanism

- Architecture: `training/energy/mobilenet_energy_pose.yaml`
- Training entry point: `training/energy/train_energy.py`
- Dataset template: `training/energy/data.yaml.example`

The model uses a MobileNetV3-based backbone with a custom `EnergyPoseDetect`
head and predicts one class with four 2D keypoints.

```bash
python training/energy/train_energy.py --data training/energy/data.yaml
```

## Armor auto-aim

- Architecture: `training/armor/armor11_corners_precision_pose.yaml`
- Training entry point: `training/armor/train_armor_precision.py`
- Dataset template: `training/armor/data.yaml.example`

This is the accuracy-first YOLO11-L model, with P3/P4/P5 detection scales and
an `ArmorPoseDetect` head. It predicts one class with four corner keypoints
(x, y, visibility).

```bash
python training/armor/train_armor_precision.py --data training/armor/data.yaml
```

The scripts accept options such as `--epochs`, `--batch`, `--device`,
`--workers`, and `--name`; outputs are written under `runs/`.

## Published weights

- `models/energy_mechanism_best.pt`: energy-mechanism pose model.
- `models/armor_precision_best.pt`: accuracy-first armor corner pose model.

The weight files are stored with Git LFS; install Git LFS before cloning if
you need the binary model files.
