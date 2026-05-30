#!/usr/bin/env python3
"""Train and export the advisory NeuralTargetingHead ONNX model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data_gen.generate_targeting_data import TargetingDataConfig, save_dataset
from training.neural_targeting.dataset import load_neural_targeting_npz, resolve_repo_path
from training.neural_targeting.model import make_model


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for neural targeting training. Install training/requirements.txt and rerun."
        ) from exc
    return torch


def make_summary_writer(enabled: bool, log_dir: str | None):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("TensorBoard is not installed; continuing without tensorboard logging.")
        return None
    return SummaryWriter(log_dir=log_dir)


def refinement_loss(torch, prediction, target, confidence_target):
    weights = 0.35 + 0.65 * confidence_target.clamp(0.0, 1.0)
    return torch.mean(((prediction[:, :2] - target) ** 2) * weights)


def confidence_calibration_loss(torch, prediction_confidence, target_confidence):
    eps = 1e-5
    pred = prediction_confidence.clamp(eps, 1.0 - eps)
    target = target_confidence.clamp(0.0, 1.0)
    return torch.mean(-(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred)))


def near_lock_penalty_loss(torch, prediction, target):
    """Discourage large learned offsets when the synthetic label says stay quiet."""
    pred_mag = torch.linalg.norm(prediction[:, :2], dim=1)
    target_mag = torch.linalg.norm(target, dim=1)
    allowed = target_mag + 2.0
    return torch.mean(torch.relu(pred_mag - allowed) ** 2)


def output_magnitude_loss(torch, prediction):
    return torch.mean(prediction[:, :2] ** 2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the advisory neural targeting head.")
    parser.add_argument("--dataset", default="training/datasets/neural_targeting_tracks.npz")
    parser.add_argument("--output", default="neural_models/neural_targeting_head.pt")
    parser.add_argument("--onnx-output", default="neural_models/neural_targeting_head.onnx")
    parser.add_argument("--metadata", default="neural_models/neural_targeting_head.json")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--confidence-weight", type=float, default=0.18)
    parser.add_argument("--near-lock-penalty-weight", type=float, default=0.035)
    parser.add_argument("--output-magnitude-weight", type=float, default=0.0008)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", default="training/runs/neural_targeting_head")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-refinement-px", type=float, default=35.0)
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
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--detection-width", type=int, default=320)
    parser.add_argument("--detection-height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--jitter-px", type=float, default=0.35)
    parser.add_argument("--camera-shake-px", type=float, default=0.45)
    parser.add_argument("--occlusion-probability", type=float, default=0.08)
    parser.add_argument("--max-speed-px-s", type=float, default=1350.0)
    parser.add_argument("--bad-prediction-probability", type=float, default=0.14)
    return parser


def export_onnx(torch, model, output_path: Path, feature_dim: int, opset: int) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(feature_dim), dtype=torch.float32)

    # Export to the C++ runtime contract: targeting_features -> targeting_output.
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["targeting_features"],
        output_names=["targeting_output"],
        dynamic_axes={
            "targeting_features": {0: "batch"},
            "targeting_output": {0: "batch"},
        },
        do_constant_folding=True,
        export_params=True,
        opset_version=max(17, int(opset)),
        dynamo=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.exists():
        if not args.auto_generate_dataset:
            raise SystemExit(
                f"Neural targeting dataset not found: {dataset_path}. "
                "Run training/data_gen/generate_targeting_data.py first, or omit --no-auto-generate-dataset."
            )

        generated = save_dataset(
            dataset_path,
            TargetingDataConfig(
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
                max_refinement_px=max(1.0, float(args.max_refinement_px)),
                bad_prediction_probability=max(0.0, min(1.0, float(args.bad_prediction_probability))),
            ),
        )
        print(f"Auto-generated missing neural targeting dataset at {generated}")

    dataset = load_neural_targeting_npz(dataset_path)
    prediction_horizon = int(args.prediction_horizon)
    if int(dataset.metadata["prediction_horizon"]) != prediction_horizon:
        raise SystemExit(
            f"Dataset prediction horizon is {dataset.metadata['prediction_horizon']}, expected {prediction_horizon}. "
            "Regenerate the dataset or pass --prediction-horizon to match it."
        )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    x_all = torch.tensor(dataset.features, dtype=torch.float32)
    y_refinement_all = torch.tensor(dataset.refinement, dtype=torch.float32)
    y_confidence_all = torch.tensor(dataset.confidence, dtype=torch.float32)

    sample_count = x_all.shape[0]
    indices = list(range(sample_count))
    random.shuffle(indices)

    val_count = max(1, int(sample_count * max(0.01, min(0.50, args.validation_fraction))))
    if val_count >= sample_count:
        val_count = max(1, sample_count // 5)
    train_indices = torch.tensor(indices[val_count:], dtype=torch.long)
    val_indices = torch.tensor(indices[:val_count], dtype=torch.long)

    train_x = x_all.index_select(0, train_indices).to(device)
    train_refinement = y_refinement_all.index_select(0, train_indices).to(device)
    train_confidence = y_confidence_all.index_select(0, train_indices).to(device)
    val_x = x_all.index_select(0, val_indices).to(device)
    val_refinement = y_refinement_all.index_select(0, val_indices).to(device)
    val_confidence = y_confidence_all.index_select(0, val_indices).to(device)

    feature_dim = int(x_all.shape[1])
    model = make_model(
        feature_dim=feature_dim,
        hidden_size=args.hidden_size,
        max_refinement_px=args.max_refinement_px,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-7, args.learning_rate),
        weight_decay=max(0.0, args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, int(args.patience) // 2),
    )
    writer = make_summary_writer(not args.no_tensorboard, args.log_dir)
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
            by_refinement = train_refinement.index_select(0, batch_idx)
            by_confidence = train_confidence.index_select(0, batch_idx)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(bx)
            ref_loss = refinement_loss(torch, prediction, by_refinement, by_confidence)
            conf_loss = confidence_calibration_loss(torch, prediction[:, 2:3], by_confidence)
            near_loss = near_lock_penalty_loss(torch, prediction, by_refinement)
            magnitude_loss = output_magnitude_loss(torch, prediction)
            loss = (
                ref_loss
                + max(0.0, args.confidence_weight) * conf_loss
                + max(0.0, args.near_lock_penalty_weight) * near_loss
                + max(0.0, args.output_magnitude_weight) * magnitude_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_ref_loss = refinement_loss(torch, val_prediction, val_refinement, val_confidence)
            val_conf_loss = confidence_calibration_loss(torch, val_prediction[:, 2:3], val_confidence)
            val_near_loss = near_lock_penalty_loss(torch, val_prediction, val_refinement)
            val_magnitude_loss = output_magnitude_loss(torch, val_prediction)
            val_total = (
                val_ref_loss
                + max(0.0, args.confidence_weight) * val_conf_loss
                + max(0.0, args.near_lock_penalty_weight) * val_near_loss
                + max(0.0, args.output_magnitude_weight) * val_magnitude_loss
            )
            val_mae = torch.mean(torch.abs(val_prediction[:, :2] - val_refinement))
            val_conf_mae = torch.mean(torch.abs(val_prediction[:, 2:3] - val_confidence))
            val_mag = torch.mean(torch.linalg.norm(val_prediction[:, :2], dim=1))

        train_loss = total_loss / max(1, seen)
        val_scalar = float(val_total.detach().cpu())
        scheduler.step(val_scalar)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_scalar, epoch)
            writer.add_scalar("metrics/val_refinement_mae", float(val_mae.detach().cpu()), epoch)
            writer.add_scalar("metrics/val_confidence_mae", float(val_conf_mae.detach().cpu()), epoch)
            writer.add_scalar("metrics/val_output_magnitude", float(val_mag.detach().cpu()), epoch)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_scalar:.6f} "
            f"val_ref_mae={float(val_mae):.4f} val_conf_mae={float(val_conf_mae):.4f} "
            f"val_mag={float(val_mag):.4f}"
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
    model_cpu.eval()

    artifact = {
        "state_dict": model_cpu.state_dict(),
        "feature_columns": dataset.metadata["feature_columns"],
        "label_columns": dataset.metadata["label_columns"],
        "prediction_horizon": prediction_horizon,
        "feature_dim": feature_dim,
        "hidden_size": max(8, int(args.hidden_size)),
        "dropout": max(0.0, min(0.50, float(args.dropout))),
        "max_refinement_px": max(1.0, float(args.max_refinement_px)),
        "dataset": str(dataset_path),
        "samples": int(sample_count),
        "best_validation_loss": best_val,
        "input_name": "targeting_features",
        "output_name": "targeting_output",
        "input_shape": [1, feature_dim],
        "output_shape": [1, 3],
    }

    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    onnx_output = resolve_repo_path(args.onnx_output)
    export_onnx(torch, model_cpu, onnx_output, feature_dim, args.opset)

    metadata = {
        "model_path": str(output),
        "onnx_path": str(onnx_output),
        "input_name": "targeting_features",
        "output_name": "targeting_output",
        "input_shape": [1, feature_dim],
        "output_shape": [1, 3],
        "feature_columns": dataset.metadata["feature_columns"],
        "label_columns": dataset.metadata["label_columns"],
        "prediction_horizon": prediction_horizon,
        "hidden_size": max(8, int(args.hidden_size)),
        "dropout": max(0.0, min(0.50, float(args.dropout))),
        "max_refinement_px": max(1.0, float(args.max_refinement_px)),
        "dataset": str(dataset_path),
        "samples": int(sample_count),
        "best_validation_loss": best_val,
        "runtime_contract": "C++ supplies pre-scaled NeuralTargetingHead features; ONNX outputs pixel offsets plus confidence.",
    }
    metadata_path = resolve_repo_path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved neural targeting artifact to {output}")
    print(f"Exported neural targeting ONNX to {onnx_output}")
    print(f"Saved neural targeting metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
