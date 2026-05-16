# PID Governor Training

This folder is offline-only. It is for creating a small MLP that learns how to
scale PID terms and final speed; it does not send mouse movement and it is not
part of runtime until the exported ONNX model is wired into C++.

## Generate a Dataset

```powershell
python training/generate_pid_dataset.py `
  --config x64/DML/config.ini `
  --output training/data/pid_governor_dataset.csv `
  --episodes-per-profile 64 `
  --steps-per-episode 180 `
  --max-speed-multiple 5
```

The generator covers stopped targets, offset stationary targets, linear motion,
direction changes, moving-away/shrinking targets, and noisy/missed detections.
Synthetic target speed is capped at:

```text
pid_max_pixel_step * pid_actuator_hz * pid_output_scale * max_speed_multiple
```

You can add `--max-speed-px-s 3000` to impose an absolute cap if the live config
is temporarily set much higher than the speed range you want to learn.

## Train

```powershell
python training/train_pid_governor.py `
  --dataset training/data/pid_governor_dataset.csv `
  --output training/models/pid_governor.pt `
  --metadata training/models/pid_governor.json `
  --epochs 25
```

The model inputs are raw controller state features. The outputs are:

```text
label_kp_scale, label_ki_scale, label_kd_scale, label_speed_scale
```

## Export ONNX

```powershell
python training/export_pid_governor_onnx.py `
  --model training/models/pid_governor.pt `
  --output training/models/pid_governor.onnx
```

The ONNX graph includes feature normalization, so C++ can feed raw feature
values in the order listed in the generated metadata JSON.

## Enable Runtime Governor

After exporting, set these in `config.ini`:

```ini
pid_governor_enabled = true
pid_governor_model_path = training/models/pid_governor.onnx
pid_governor_blend = 1.000
pid_governor_max_speed_multiple = 5.000
```

If the file is missing or inference fails, runtime falls back to pure PID.

## Evaluate

```powershell
python training/evaluate_pid_governor.py `
  --dataset training/data/pid_governor_dataset.csv `
  --model training/models/pid_governor.pt
```
