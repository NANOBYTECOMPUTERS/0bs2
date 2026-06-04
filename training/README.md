# Neural Pipeline Training

This directory contains the complete offline training stack for the advisory neural components used by the runtime (Perfect Aim v1).

**Components covered:**
- PID Governor (learned dynamic scaling of PID gains + speed)
- Temporal Predictor (GRU-based future position forecasting)
- Neural Targeting Head (bounded aim refinement + confidence)
- Neural Tracker (learned association helper)

All trainers have been modernized with consistent practices: AdamW + weight decay, learning rate scheduling, early stopping, gradient clipping, optional TensorBoard, and (strongly recommended) LightGBM teacher distillation.

Models are centralized under `neural_models/` at the project root for easier distribution. The C++ runtime searches `neural_models/`, `models/`, and `training/models/`.

See the main [README.md](../README.md) (Neural Tab and Perfect Aim v1 sections) for runtime configuration and the advisory nature of these systems.

---

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

The model inputs are raw controller state features (40 columns in the exact order emitted by `pidGovernorFeatures()` in C++). The outputs are:

```text
label_kp_scale, label_ki_scale, label_kd_scale, label_speed_scale
```

**Important (2025+):** The PID governor trainer and dataset were updated to the current 40-feature runtime contract (added target_offset_*, aim_point_error_*, box_aspect_ratio, plus refined naming). Regenerate your dataset with `generate_pid_dataset.py` (or the vendored copy) after pulling these changes, then retrain.

## Export ONNX

```powershell
python training/export_pid_governor_onnx.py `
  --model neural_models/pid_governor.pt `
  --output neural_models/pid_governor.onnx
```

All neural models are now centralized in the `neural_models/` folder at the project root
for easier management and distribution. The C++ runtime automatically searches
`neural_models/`, `models/`, and `training/models/` when resolving model paths.

## Enable Runtime Governor

After exporting, set these in `config.ini`:

```ini
pid_governor_enabled = true
pid_governor_model_path = training/models/pid_governor.onnx
pid_governor_blend = 1.000
pid_governor_max_speed_multiple = 5.000
```

If the file is missing or inference fails, runtime falls back to pure PID.

**Training hygiene:** All four trainers now share modern practices (AdamW + weight decay, ReduceLROnPlateau or equivalent, early stopping + best checkpoint restore, gradient clipping, optional TensorBoard). The two previously lighter trainers (neural_tracker, pid_governor) received the largest upgrades.

**Strongly recommended:** Use `--use-gbm-teacher` (plus `--distill-weight 0.5` to `0.7`) when training the PID Governor or Neural Tracker. This trains a LightGBM teacher on the same data and distills its predictions into the neural net. The final exported ONNX remains a small, smooth MLP (zero changes to the C++ runtime), but the weights are significantly better. LightGBM must be installed (`pip install lightgbm`).

## Evaluate

```powershell
python training/evaluate_pid_governor.py `
  --dataset training/data/pid_governor_dataset.csv `
  --model training/models/pid_governor.pt
```

# Neural Tracker Training

This trains the optional association helper used by `MultiTargetTracker`. The
runtime still owns gating and convergence; the neural tracker only adds a small
advisory score bonus/penalty to candidate association.

## Generate Association Data

```powershell
python training/generate_neural_tracker_dataset.py `
  --output training/data/neural_tracker_dataset.csv `
  --samples 20000
```

Synthetic generation now writes grouped candidate pairs. Each group contains a
true match plus at least one hard distractor when possible, including near
duplicates, pivot-offset errors, heading conflicts, size mismatches, and allowed
head/body swap compatibility (`class_compatible = 0.5`). Runtime logs remain an
optional add-on:

```powershell
python training/generate_neural_tracker_dataset.py `
  --merge-log training/logs/neural_tracker_association.csv
```

## Train And Export

```powershell
python training/train_neural_tracker.py `
  --dataset training/data/neural_tracker_dataset.csv `
  --output neural_models/neural_tracker.pt `
  --metadata neural_models/neural_tracker.json `
  --use-gbm-teacher `
  --distill-weight 0.20 `
  --ranking-loss-weight 0.20 `
  --learning-rate 0.00045 `
  --weight-decay 0.00012 `
  --epochs 30 `
  --patience 8

python training/export_neural_tracker_onnx.py `
  --model neural_models/neural_tracker.pt `
  --output neural_models/neural_tracker.onnx
```

The trainer uses grouped train/validation splitting, optional train-only
LightGBM distillation, pairwise ranking loss, and validation-set temperature
calibration. Real-world/log rows are optional and disabled by default:

```powershell
python training/train_neural_tracker.py `
  --dataset training/data/neural_tracker_dataset.csv `
  --real-world-data training/logs/neural_tracker_association.csv
```

## Evaluate

```powershell
python training/evaluate_neural_tracker.py `
  --dataset training/data/neural_tracker_dataset.csv `
  --model neural_models/neural_tracker.pt `
  --output training/models/neural_tracker_eval.json
```

Prefer grouped metrics over plain threshold accuracy. The evaluator reports
weighted BCE, Brier score, ECE calibration error, ROC/PR AUC, and grouped top-1
association accuracy.

# Temporal Predictor Training

This trains the learned future-position predictor used by the optional C++
`TemporalPredictor` path. The exported ONNX accepts raw runtime history:

```text
[batch, history_length, 8]
[x, y, w, h, vx, vy, box_scale_vel, confidence]
```

and returns:

```text
[batch, prediction_horizon, 2]
[future_x, future_y]
```

## Generate Track Data

```powershell
python training/data_gen/generate_track_data.py `
  --output training/datasets/temporal_tracks.npz `
  --samples 8192 `
  --history-length 12 `
  --prediction-horizon 16 `
  --detection-width 320 `
  --detection-height 320
```

The generator covers constant velocity, acceleration, curves, abrupt direction
changes, stop-and-go motion, partial occlusion, jitter, and camera shake.

## Train And Export

```powershell
python training/train_temporal.py `
  --dataset training/datasets/temporal_tracks_v2.npz `
  --output training/models/temporal_predictor.pt `
  --onnx-output models/temporal_predictor.onnx `
  --metadata training/models/temporal_predictor.json `
  --model-type gru `
  --hidden-size 160 `
  --auto-generate-samples 65536 `
  --max-speed-px-s 1800 `
  --jitter-px 0.55 `
  --camera-shake-px 0.75 `
  --occlusion-probability 0.12 `
  --near-horizon-frames 8 `
  --near-horizon-weight 3.5 `
  --latency-focus-start-frame 3 `
  --latency-focus-end-frame 6 `
  --latency-focus-weight 0.75 `
  --velocity-weight 0.10 `
  --smoothness-weight 0.008 `
  --epochs 60 `
  --patience 10
```

If the dataset path does not exist, `train_temporal.py` auto-generates it using
the requested history length, horizon, and detection resolution. Add
`--no-auto-generate-dataset` when you want a missing dataset to be treated as an
error.

`--model-type transformer` is available for experiments, but the GRU is the
default because it is small, fast, and matches the runtime latency goal.
The default loss weights the first six predicted frames more heavily than the
far end of the horizon, which makes the exported model more useful for PID
feed-forward and less likely to over-optimize distant positions.
The latency-focus loss additionally weights frames 3-6 by default, matching the
age-compensated runtime feed-forward point used when async predictions are a few
frames old.

## Evaluate

```powershell
python training/evaluate.py `
  --dataset training/datasets/temporal_tracks.npz `
  --model training/models/temporal_predictor.pt `
  --output-dir training/models/temporal_eval
```

Evaluation writes `metrics.json` with ADE, FDE, and smoothness score, plus
latency-frame ADE and trajectory comparison plots when `matplotlib` is
installed.

## Runtime Config

After export, enable the runtime path with:

```ini
temporal_prediction_enabled = true
temporal_prediction_model_path = models/temporal_predictor.onnx
temporal_prediction_history_length = 12
temporal_prediction_horizon = 16
temporal_prediction_interval_frames = 1
```

Training-only knobs such as `hidden_size` and `model_type` are CLI arguments,
not runtime fields. The runtime only needs the exported ONNX contract and the
history/horizon values used during export.

# Neural Targeting Head Training

This trains the optional advisory `NeuralTargetingHead`. It does not replace
PID/Kalman convergence. The exported ONNX only predicts a bounded feed-forward
refinement offset and confidence:

```text
input:  targeting_features [batch, 10 + prediction_horizon * 2]
output: targeting_output   [batch, 3]
        [refinement_offset_x, refinement_offset_y, confidence]
```

The feature vector intentionally matches `neural/targeting/NeuralTargetingHead.cpp`.
C++ already supplies scaled values, so this trainer does not wrap the model with
an additional mean/std normalizer.

## Generate Targeting Data

```powershell
python training/data_gen/generate_targeting_data.py `
  --output training/datasets/neural_targeting_tracks.npz `
  --samples 8192 `
  --history-length 12 `
  --prediction-horizon 16 `
  --detection-width 320 `
  --detection-height 320
```

The generator builds on the temporal track simulator, then adds realistic
prediction bias, bad-prediction cases, near-lock damping, and ideal refinement
labels. Labels are clamped to the configured maximum refinement so the model
learns the same advisory boundary enforced at runtime.

## Train And Export

```powershell
python training/train_neural_targeting.py `
  --dataset training/datasets/neural_targeting_tracks.npz `
  --output training/models/neural_targeting_head.pt `
  --onnx-output models/neural_targeting_head.onnx `
  --metadata training/models/neural_targeting_head.json `
  --prediction-horizon 16 `
  --hidden-size 192 `
  --max-refinement-px 35 `
  --epochs 35
```

If the dataset path does not exist, `train_neural_targeting.py` auto-generates
it using the requested horizon and detection geometry. Add
`--no-auto-generate-dataset` when a missing dataset should fail.

## Evaluate

```powershell
python training/evaluate_neural_targeting.py `
  --dataset training/datasets/neural_targeting_tracks.npz `
  --model training/models/neural_targeting_head.pt `
  --output-dir training/models/neural_targeting_eval
```

Evaluation writes `metrics.json` with refinement MAE/RMSE, confidence error, and
bad-overconfidence rate. If `matplotlib` is installed, it also writes plots
showing the current point, predicted trajectory, ideal refinement, and model
refinement.
