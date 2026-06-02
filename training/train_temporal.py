#!/usr/bin/env python3
"""Train and export the learned temporal predictor."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data_gen.generate_track_data import TrackDataConfig, save_dataset
from training.temporal_predictor.dataset import FEATURE_COLUMNS, TemporalDataset, load_temporal_npz, resolve_repo_path
from training.temporal_predictor.model import NormalizedTemporalPredictor, make_model


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for temporal predictor training. Install training/requirements.txt and rerun."
        ) from exc
    return torch


def make_summary_writer(torch, enabled: bool, log_dir: str | None):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("TensorBoard is not installed; continuing without tensorboard logging.")
        return None
    return SummaryWriter(log_dir=log_dir)


def smoothness_loss(torch, prediction):
    if prediction.shape[1] < 3:
        return prediction.sum() * 0.0
    deltas = prediction[:, 1:, :] - prediction[:, :-1, :]
    second_delta = deltas[:, 1:, :] - deltas[:, :-1, :]
    return torch.mean(second_delta * second_delta)


def velocity_consistency_loss(torch, prediction, target):
    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0
    pred_delta = prediction[:, 1:, :] - prediction[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]
    return torch.mean((pred_delta - target_delta) ** 2)


def build_horizon_weights(torch, horizon: int, near_frames: int, near_weight: float, device):
    horizon = max(1, int(horizon))
    near_frames = max(0, min(horizon, int(near_frames)))
    weights = torch.ones(horizon, dtype=torch.float32, device=device)
    if near_frames > 0 and near_weight > 1.0:
        weights[:near_frames] = float(near_weight)
    return weights / weights.mean().clamp_min(1e-6)


def weighted_position_loss(torch, prediction, target, horizon_weights):
    squared = (prediction - target) ** 2
    weights = horizon_weights.reshape(1, -1, 1).to(device=prediction.device, dtype=prediction.dtype)
    return torch.mean(squared * weights)


def arg_supplied(argv_tokens: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv_tokens)


def _profiles(dataset: TemporalDataset, fallback: str) -> np.ndarray:
    if dataset.profiles.shape[0] == dataset.history.shape[0]:
        return dataset.profiles
    return np.full((dataset.history.shape[0],), fallback, dtype="<U24")


def mix_temporal_datasets(
    synthetic: TemporalDataset,
    real_world: TemporalDataset,
    mix_ratio: float,
    seed: int,
) -> TemporalDataset:
    if synthetic.history.shape[1:] != real_world.history.shape[1:]:
        raise ValueError("real-world temporal history shape does not match the training dataset")
    if synthetic.future.shape[1:] != real_world.future.shape[1:]:
        raise ValueError("real-world temporal future shape does not match the training dataset")

    ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return synthetic
    if ratio >= 1.0:
        metadata = dict(real_world.metadata)
        metadata.update({"source": "real_world", "mix_ratio_real": 1.0, "samples": int(real_world.history.shape[0])})
        return TemporalDataset(
            history=real_world.history,
            future=real_world.future,
            profiles=_profiles(real_world, "real_world"),
            metadata=metadata,
        )

    desired_synthetic = int(round(real_world.history.shape[0] * (1.0 - ratio) / max(1e-6, ratio)))
    desired_synthetic = max(1, min(desired_synthetic, synthetic.history.shape[0]))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(synthetic.history.shape[0], size=desired_synthetic, replace=False)
    history = np.concatenate([real_world.history, synthetic.history[idx]], axis=0).astype(np.float32, copy=False)
    future = np.concatenate([real_world.future, synthetic.future[idx]], axis=0).astype(np.float32, copy=False)
    profiles = np.concatenate([_profiles(real_world, "real_world"), _profiles(synthetic, "synthetic")[idx]], axis=0)
    metadata = dict(synthetic.metadata)
    metadata.update(
        {
            "source": "mixed_real_world",
            "real_world_samples": int(real_world.history.shape[0]),
            "synthetic_samples": int(desired_synthetic),
            "mix_ratio_real": ratio,
            "samples": int(history.shape[0]),
        }
    )
    return TemporalDataset(history=history, future=future, profiles=profiles, metadata=metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned temporal predictor.")
    parser.add_argument("--dataset", default="training/datasets/temporal_tracks.npz")
    parser.add_argument("--output", default="neural_models/temporal_predictor.pt")
    parser.add_argument("--onnx-output", default="neural_models/temporal_predictor.onnx")
    parser.add_argument("--metadata", default="neural_models/temporal_predictor.json")
    parser.add_argument("--fine-tune", action="store_true", help="Continue training from an existing checkpoint.")
    parser.add_argument("--checkpoint", default="neural_models/temporal_predictor.pt")
    parser.add_argument("--real-world-data", default=None, help="Optional converted real-world temporal .npz dataset.")
    parser.add_argument("--mix-ratio", type=float, default=0.40, help="Fraction of training samples to draw from real-world data.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--model-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.015)
    parser.add_argument("--velocity-weight", type=float, default=0.050)
    parser.add_argument("--near-horizon-frames", type=int, default=6)
    parser.add_argument("--near-horizon-weight", type=float, default=2.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", default="training/runs/temporal_predictor")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--auto-generate-dataset",
        dest="auto_generate_dataset",
        action="store_true",
        default=True,
        help="Generate the dataset if --dataset does not exist.",
    )
    parser.add_argument(
        "--no-auto-generate-dataset",
        dest="auto_generate_dataset",
        action="store_false",
        help="Fail if --dataset does not exist instead of generating it.",
    )
    parser.add_argument("--auto-generate-samples", type=int, default=8192)
    parser.add_argument("--detection-width", type=int, default=320)
    parser.add_argument("--detection-height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--jitter-px", type=float, default=0.35)
    parser.add_argument("--camera-shake-px", type=float, default=0.45)
    parser.add_argument("--occlusion-probability", type=float, default=0.08)
    parser.add_argument("--max-speed-px-s", type=float, default=1350.0)
    return parser


def export_onnx(torch, model, artifact: dict, output_path: Path, opset: int) -> None:
    wrapped = NormalizedTemporalPredictor(
        model,
        artifact["feature_mean"],
        artifact["feature_std"],
    )
    wrapped.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(
        1,
        int(artifact["history_length"]),
        len(artifact["feature_columns"]),
        dtype=torch.float32,
    )

    # Export to the C++ runtime contract: temporal_history -> future_xy.
    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["temporal_history"],
        output_names=["future_xy"],
        dynamic_axes={
            "temporal_history": {0: "batch"},
            "future_xy": {0: "batch"},
        },
        do_constant_folding=True,
        export_params=True,
        opset_version=max(17, int(opset)),
        dynamo=False,
    )


def main(argv: list[str] | None = None) -> int:
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    checkpoint_artifact = None
    if args.fine_tune:
        if not arg_supplied(argv_tokens, "--output"):
            args.output = "neural_models/temporal_predictor_realworld.pt"
        if not arg_supplied(argv_tokens, "--onnx-output"):
            args.onnx_output = "neural_models/temporal_predictor_realworld.onnx"  # realworld.onnx
        if not arg_supplied(argv_tokens, "--metadata"):
            args.metadata = "neural_models/temporal_predictor_realworld.json"
        if not arg_supplied(argv_tokens, "--learning-rate"):
            args.learning_rate = min(float(args.learning_rate), 1e-4)

        checkpoint_path = resolve_repo_path(args.checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"Fine-tune checkpoint not found: {checkpoint_path}")
        checkpoint_artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not arg_supplied(argv_tokens, "--history-length"):
            args.history_length = int(checkpoint_artifact.get("history_length", args.history_length))
        if not arg_supplied(argv_tokens, "--prediction-horizon"):
            args.prediction_horizon = int(checkpoint_artifact.get("prediction_horizon", args.prediction_horizon))
        if not arg_supplied(argv_tokens, "--hidden-size"):
            args.hidden_size = int(checkpoint_artifact.get("hidden_size", args.hidden_size))
        if not arg_supplied(argv_tokens, "--model-type"):
            args.model_type = str(checkpoint_artifact.get("model_type", args.model_type))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.exists():
        if not args.auto_generate_dataset:
            raise SystemExit(
                f"Temporal dataset not found: {dataset_path}. "
                "Run training/data_gen/generate_track_data.py first, or omit --no-auto-generate-dataset."
            )

        generated = save_dataset(
            dataset_path,
            TrackDataConfig(
                samples=max(2, int(args.auto_generate_samples)),
                history_length=max(2, int(args.history_length)),
                prediction_horizon=max(1, int(args.prediction_horizon)),
                detection_width=max(8, int(args.detection_width)),
                detection_height=max(8, int(args.detection_height)),
                fps=max(1.0, float(args.fps)),
                seed=int(args.seed),
                jitter_px=max(0.0, float(args.jitter_px)),
                camera_shake_px=max(0.0, float(args.camera_shake_px)),
                occlusion_probability=max(0.0, min(1.0, float(args.occlusion_probability))),
                max_speed_px_s=max(1.0, float(args.max_speed_px_s)),
            ),
        )
        print(f"Auto-generated missing temporal dataset at {generated}")

    dataset = load_temporal_npz(dataset_path)
    if args.real_world_data:
        real_world_dataset = load_temporal_npz(args.real_world_data)
        dataset = mix_temporal_datasets(dataset, real_world_dataset, args.mix_ratio, args.seed)

    history_length = int(args.history_length)
    prediction_horizon = int(args.prediction_horizon)
    if dataset.history.shape[1] != history_length:
        raise SystemExit(
            f"Dataset history length is {dataset.history.shape[1]}, expected {history_length}. "
            "Regenerate the dataset or pass --history-length to match it."
        )
    if dataset.future.shape[1] != prediction_horizon:
        raise SystemExit(
            f"Dataset prediction horizon is {dataset.future.shape[1]}, expected {prediction_horizon}. "
            "Regenerate the dataset or pass --prediction-horizon to match it."
        )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    x_all = torch.tensor(dataset.history, dtype=torch.float32)
    y_all = torch.tensor(dataset.future, dtype=torch.float32)
    sample_count = x_all.shape[0]
    indices = list(range(sample_count))
    random.shuffle(indices)

    val_count = max(1, int(sample_count * max(0.01, min(0.50, args.validation_fraction))))
    if val_count >= sample_count:
        val_count = max(1, sample_count // 5)
    train_indices = torch.tensor(indices[val_count:], dtype=torch.long)
    val_indices = torch.tensor(indices[:val_count], dtype=torch.long)

    train_x_raw = x_all.index_select(0, train_indices)
    train_y = y_all.index_select(0, train_indices)
    val_x_raw = x_all.index_select(0, val_indices)
    val_y = y_all.index_select(0, val_indices)

    feature_mean = train_x_raw.reshape(-1, len(FEATURE_COLUMNS)).mean(dim=0)
    feature_std = train_x_raw.reshape(-1, len(FEATURE_COLUMNS)).std(dim=0).clamp_min(1e-6)
    train_x = (train_x_raw - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)
    val_x = (val_x_raw - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    model = make_model(
        feature_dim=len(FEATURE_COLUMNS),
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        hidden_size=args.hidden_size,
        model_type=args.model_type,
    ).to(device)
    if checkpoint_artifact is not None:
        model.load_state_dict(checkpoint_artifact["state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-7, args.learning_rate),
        weight_decay=max(0.0, args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )
    writer = make_summary_writer(torch, not args.no_tensorboard, args.log_dir)
    horizon_weights = build_horizon_weights(
        torch,
        prediction_horizon,
        args.near_horizon_frames,
        args.near_horizon_weight,
        device,
    )
    batch_size = max(4, int(args.batch_size))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], device=device)
        total_loss = 0.0
        seen = 0
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            bx = train_x.index_select(0, batch_idx)
            by = train_y.index_select(0, batch_idx)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bx)
            main_loss = weighted_position_loss(torch, prediction, by, horizon_weights)
            smooth_loss = smoothness_loss(torch, prediction)
            velocity_loss = velocity_consistency_loss(torch, prediction, by)
            loss = (
                main_loss
                + max(0.0, args.smoothness_weight) * smooth_loss
                + max(0.0, args.velocity_weight) * velocity_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_mse = weighted_position_loss(torch, val_prediction, val_y, horizon_weights)
            near_frames = max(1, min(prediction_horizon, int(args.near_horizon_frames)))
            val_near_ade = torch.mean(
                torch.linalg.norm(
                    val_prediction[:, :near_frames, :] - val_y[:, :near_frames, :],
                    dim=-1,
                )
            )
            val_ade = torch.mean(torch.linalg.norm(val_prediction - val_y, dim=-1))
            val_fde = torch.mean(torch.linalg.norm(val_prediction[:, -1, :] - val_y[:, -1, :], dim=-1))
            val_total = (
                val_mse
                + max(0.0, args.smoothness_weight) * smoothness_loss(torch, val_prediction)
                + max(0.0, args.velocity_weight) * velocity_consistency_loss(torch, val_prediction, val_y)
            )

        train_loss = total_loss / max(1, seen)
        val_scalar = float(val_total.detach().cpu())
        scheduler.step(val_scalar)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_scalar, epoch)
            writer.add_scalar("metrics/val_near_ade", float(val_near_ade.detach().cpu()), epoch)
            writer.add_scalar("metrics/val_ade", float(val_ade.detach().cpu()), epoch)
            writer.add_scalar("metrics/val_fde", float(val_fde.detach().cpu()), epoch)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_scalar:.6f} val_near_ade={float(val_near_ade):.4f} "
            f"val_ade={float(val_ade):.4f} val_fde={float(val_fde):.4f}"
        )

        if val_scalar < best_val - 1e-6:
            best_val = val_scalar
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(args.patience)):
                print("early stopping: validation loss did not improve")
                break

    if writer is not None:
        writer.close()

    model.load_state_dict(best_state)
    model_cpu = model.to("cpu")
    artifact = {
        "state_dict": model_cpu.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "history_length": history_length,
        "prediction_horizon": prediction_horizon,
        "feature_dim": len(FEATURE_COLUMNS),
        "hidden_size": max(8, int(args.hidden_size)),
        "model_type": args.model_type,
        "near_horizon_frames": max(1, min(prediction_horizon, int(args.near_horizon_frames))),
        "near_horizon_weight": max(1.0, float(args.near_horizon_weight)),
        "dataset": str(resolve_repo_path(args.dataset)),
        "real_world_data": str(resolve_repo_path(args.real_world_data)) if args.real_world_data else "",
        "fine_tuned": bool(args.fine_tune),
        "checkpoint": str(resolve_repo_path(args.checkpoint)) if args.fine_tune else "",
        "mix_ratio_real": float(np.clip(args.mix_ratio, 0.0, 1.0)) if args.real_world_data else 0.0,
        "samples": int(sample_count),
        "fps": float(dataset.metadata.get("fps", 60.0)),
        "best_validation_loss": best_val,
    }

    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    onnx_output = resolve_repo_path(args.onnx_output)
    export_onnx(torch, model_cpu, artifact, onnx_output, args.opset)

    metadata = {
        "model_path": str(output),
        "onnx_path": str(onnx_output),
        "input_name": "temporal_history",
        "output_name": "future_xy",
        "input_shape": [1, history_length, len(FEATURE_COLUMNS)],
        "output_shape": [1, prediction_horizon, 2],
        "feature_columns": FEATURE_COLUMNS,
        "history_length": history_length,
        "prediction_horizon": prediction_horizon,
        "hidden_size": max(8, int(args.hidden_size)),
        "model_type": args.model_type,
        "near_horizon_frames": max(1, min(prediction_horizon, int(args.near_horizon_frames))),
        "near_horizon_weight": max(1.0, float(args.near_horizon_weight)),
        "best_validation_loss": best_val,
        "dataset": str(resolve_repo_path(args.dataset)),
        "real_world_data": str(resolve_repo_path(args.real_world_data)) if args.real_world_data else "",
        "fine_tuned": bool(args.fine_tune),
        "checkpoint": str(resolve_repo_path(args.checkpoint)) if args.fine_tune else "",
        "mix_ratio_real": float(np.clip(args.mix_ratio, 0.0, 1.0)) if args.real_world_data else 0.0,
        "samples": int(sample_count),
    }
    metadata_path = resolve_repo_path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved temporal predictor artifact to {output}")
    print(f"Exported temporal predictor ONNX to {onnx_output}")
    print(f"Saved temporal predictor metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
