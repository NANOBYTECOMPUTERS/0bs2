#!/usr/bin/env python3
"""Train a small MLP that outputs PID gain/speed scales."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path


def ensure_lightgbm():
    """Auto-install lightgbm when --use-gbm-teacher is requested."""
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        print("\n[Dependency] lightgbm is required for --use-gbm-teacher.")
        answer = input("Install it now? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
                print("lightgbm installed successfully.\n")
                return True
            except Exception as e:
                print(f"Failed to auto-install lightgbm: {e}")
                print("Please run: pip install lightgbm\n")
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pid_governor.dataset import FEATURE_COLUMNS, LABEL_COLUMNS, read_dataset


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install it in your Python environment, "
            "then rerun this script."
        ) from exc
    return torch


def make_model(torch, input_dim: int, output_dim: int, hidden: int):
    class PidGovernorNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, output_dim),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    return PidGovernorNet()


def rows_to_tensors(torch, rows):
    features = [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]
    labels = [[float(row[column]) for column in LABEL_COLUMNS] for row in rows]
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


def try_train_gbm_teacher(rows, feature_columns, label_columns, args, mean, std):
    """Train one LightGBM regressor per PID output and stack soft targets."""
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        print("LightGBM not installed. Falling back to pure neural training. "
              "Install with: pip install lightgbm")
        return None

    print(f"Training LightGBM teacher for PID Governor ({args.gbm_boosting_rounds} rounds)...")
    X = np.array([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float32)
    X = (X - mean.numpy()) / std.numpy()
    y_matrix = np.array([[float(row[col]) for col in label_columns] for row in rows], dtype=np.float32)

    gbm_models = []
    soft_columns = []
    for output_index, label_column in enumerate(label_columns):
        train_data = lgb.Dataset(X, label=y_matrix[:, output_index])
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": args.gbm_learning_rate,
            "max_depth": args.gbm_max_depth,
            "num_leaves": 63,
            "verbose": -1,
            "seed": args.seed + output_index,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=args.gbm_boosting_rounds,
        )
        prediction = np.asarray(model.predict(X), dtype=np.float32)
        soft_columns.append(np.clip(prediction, 0.0, 1.0))
        gbm_models.append(model)
        print(f"  trained teacher output: {label_column}")

    soft_targets = np.stack(soft_columns, axis=1)
    print(f"  GBM teacher trained. Soft target shape: {soft_targets.shape}")
    return gbm_models, soft_targets.astype(np.float32)


def make_summary_writer(enabled: bool, log_dir: str | None):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("TensorBoard is not installed; continuing without tensorboard logging.")
        return None
    return SummaryWriter(log_dir=log_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PID governor MLP (40-feature contract).")
    parser.add_argument("--dataset", default="training/data/pid_governor_dataset.csv")
    parser.add_argument("--output", default="neural_models/pid_governor.pt")
    parser.add_argument("--metadata", default="neural_models/pid_governor.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=72)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=1.5e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", default="training/runs/pid_governor")
    parser.add_argument("--no-tensorboard", action="store_true")
    # GBM teacher + distillation (recommended upgrade for tabular regression)
    parser.add_argument("--use-gbm-teacher", action="store_true",
                        help="Train a LightGBM regressor teacher and distill its predictions into the neural net.")
    parser.add_argument("--gbm-boosting-rounds", type=int, default=300)
    parser.add_argument("--gbm-learning-rate", type=float, default=0.04)
    parser.add_argument("--gbm-max-depth", type=int, default=7)
    parser.add_argument("--distill-weight", type=float, default=0.55,
                        help="Weight given to GBM soft targets vs ground-truth labels (0.0 = no distillation).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    dataset_path = resolve_repo_path(args.dataset)
    rows = read_dataset(dataset_path)
    if not rows:
        raise SystemExit(f"No rows found in {dataset_path}")
    missing = [column for column in FEATURE_COLUMNS + LABEL_COLUMNS if column not in rows[0]]
    if missing:
        missing_preview = ", ".join(missing[:8])
        raise SystemExit(
            f"Dataset {dataset_path} is missing required columns: {missing_preview}. "
            "Regenerate it with generate_pid_dataset.py before training."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    # Build tensors with tolerance for old datasets (fill missing new columns with 0.0)
    def _get_or_default(row, col, default=0.0):
        if col in row:
            v = float(row[col])
            return v if math.isfinite(v) else default
        return default

    features = []
    labels = []
    for row in rows:
        feat = [_get_or_default(row, c, 0.0) for c in FEATURE_COLUMNS]
        lab = [float(row[c]) for c in LABEL_COLUMNS]
        features.append(feat)
        labels.append(lab)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    # Optional GBM teacher + distillation for the 40-feature governor
    gbm_model = None
    gbm_soft_targets = None
    if getattr(args, "use_gbm_teacher", False):
        if ensure_lightgbm():
            gbm_result = try_train_gbm_teacher(rows, FEATURE_COLUMNS, LABEL_COLUMNS, args, mean, std)
        else:
            print("Continuing without GBM teacher.")
        if gbm_result is not None:
            gbm_model, gbm_soft_np = gbm_result
            gbm_soft_targets = torch.tensor(gbm_soft_np, dtype=torch.float32)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    val_count = max(1, int(len(indices) * max(0.01, min(0.50, args.validation_fraction))))
    if val_count >= len(indices):
        val_count = max(1, len(indices) // 5)
    val_idx = torch.tensor(indices[:val_count], dtype=torch.long)
    train_idx = torch.tensor(indices[val_count:], dtype=torch.long)

    train_x = x.index_select(0, train_idx).to(device)
    train_y = y.index_select(0, train_idx).to(device)
    val_x = x.index_select(0, val_idx).to(device)
    val_y = y.index_select(0, val_idx).to(device)

    if gbm_soft_targets is not None:
        train_gbm_soft = gbm_soft_targets.index_select(0, train_idx).to(device)
        val_gbm_soft = gbm_soft_targets.index_select(0, val_idx).to(device)
    else:
        train_gbm_soft = val_gbm_soft = None

    hidden = max(16, int(args.hidden))
    model = make_model(torch, len(FEATURE_COLUMNS), len(LABEL_COLUMNS), hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-7, args.learning_rate),
        weight_decay=max(0.0, args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 2)
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = max(16, int(args.batch_size))
    writer = make_summary_writer(not args.no_tensorboard, args.log_dir)

    distill_w = float(max(0.0, min(0.9, args.distill_weight)))
    use_distill = (train_gbm_soft is not None) and (distill_w > 0.0)

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
            by_hard = train_y.index_select(0, batch_idx)

            # Distillation: blend hard teacher labels with GBM soft predictions
            if use_distill:
                by = (1.0 - distill_w) * by_hard + distill_w * train_gbm_soft.index_select(0, batch_idx)
            else:
                by = by_hard

            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            # Always report against hard labels
            val_loss = float(loss_fn(val_pred, val_y).item())
            val_mae = float(torch.mean(torch.abs(val_pred - val_y)).item())
            val_max = float(torch.max(torch.abs(val_pred - val_y)).item())

        train_loss = total_loss / max(1, seen)
        scheduler.step(val_loss)

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/val_mae", val_mae, epoch)
            writer.add_scalar("metrics/val_max_abs", val_max, epoch)

        mode = " (distilled)" if use_distill else ""
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
            f"val_mae={val_mae:.5f} val_max={val_max:.4f}{mode}"
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
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

    output_path = resolve_repo_path(args.output)
    metadata_path = resolve_repo_path(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "state_dict": model_cpu.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "hidden": hidden,
        "dataset": str(dataset_path),
        "samples": len(rows),
        "best_validation_loss": best_val,
        "input_name": "pid_features",
        "output_name": "pid_scales",
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
    }
    torch.save(artifact, output_path)

    metadata = {
        "model_path": str(output_path),
        "dataset": str(dataset_path),
        "samples": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "hidden": hidden,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "best_validation_mse": best_val,
        "final_validation_mae": val_mae,
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
        "runtime_contract": "C++ feeds 40 raw features (exact order in feature_columns); ONNX applies normalization internally and outputs 4 sigmoid scales.",
    }
    if gbm_model is not None:
        metadata["gbm_teacher"] = {
            "boosting_rounds": args.gbm_boosting_rounds,
            "learning_rate": args.gbm_learning_rate,
            "max_depth": args.gbm_max_depth,
        }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved PID governor artifact to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    if use_distill:
        print("  (Model was improved via LightGBM distillation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
