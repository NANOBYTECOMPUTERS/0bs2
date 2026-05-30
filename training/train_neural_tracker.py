#!/usr/bin/env python3
"""Train the optional neural tracker association MLP."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path


def ensure_lightgbm():
    """Auto-install lightgbm if the user requested GBM distillation."""
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        print("\n[Dependency] lightgbm is required for --use-gbm-teacher but is not installed.")
        answer = input("Install lightgbm now? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
                print("lightgbm installed successfully.\n")
                return True
            except Exception as e:
                print(f"Failed to install lightgbm: {e}")
                print("Please install manually: pip install lightgbm\n")
        return False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.neural_tracker.dataset import FEATURE_COLUMNS, LABEL_COLUMN, read_dataset


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for neural tracker training. Install torch and rerun.") from exc
    return torch


def make_model(torch, input_dim: int, hidden: int):
    class NeuralTrackerNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.035),
                torch.nn.Linear(hidden, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    return NeuralTrackerNet()


def rows_to_tensors(torch, rows):
    features = [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]
    labels = [[float(row[LABEL_COLUMN])] for row in rows]
    weights = [[float(row.get("sample_weight", 1.0))] for row in rows]
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32).clamp_min(0.05)
    return x, y, w


def try_train_gbm_teacher(rows, feature_columns, label_column, args, mean, std):
    """Train a LightGBM teacher if available. Returns soft probabilities or None."""
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        print("LightGBM not installed. Falling back to pure neural training. "
              "Install with: pip install lightgbm")
        return None

    print(f"Training LightGBM teacher ({args.gbm_boosting_rounds} rounds)...")
    X = np.array([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float32)
    X = (X - mean.numpy()) / std.numpy()   # use the same normalization as the NN
    y = np.array([float(row[label_column]) for row in rows], dtype=np.float32)
    w = np.array([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32).clip(0.05, None)

    train_data = lgb.Dataset(X, label=y, weight=w)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.gbm_learning_rate,
        "max_depth": args.gbm_max_depth,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": args.seed,
    }

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=args.gbm_boosting_rounds,
    )

    soft_probs = gbm.predict(X, raw_score=False)
    print(f"  GBM teacher trained. Example soft prob range: [{soft_probs.min():.3f}, {soft_probs.max():.3f}]")

    return gbm, soft_probs.astype(np.float32)


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
    parser = argparse.ArgumentParser(description="Train the neural tracker association model.")
    parser.add_argument("--dataset", default="training/data/neural_tracker_dataset.csv")
    parser.add_argument("--output", default="neural_models/neural_tracker.pt")
    parser.add_argument("--metadata", default="neural_models/neural_tracker.json")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=56)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-dir", default="training/runs/neural_tracker")
    parser.add_argument("--no-tensorboard", action="store_true")
    # GBM teacher + distillation (recommended upgrade for tabular association)
    parser.add_argument("--use-gbm-teacher", action="store_true",
                        help="Train a LightGBM teacher and distill its soft predictions into the neural net.")
    parser.add_argument("--gbm-boosting-rounds", type=int, default=250)
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbm-max-depth", type=int, default=6)
    parser.add_argument("--distill-weight", type=float, default=0.65,
                        help="Weight given to GBM soft labels vs hard labels during NN training (0.0 = pure hard labels).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    dataset_path = resolve_repo_path(args.dataset)
    rows = read_dataset(dataset_path)
    if not rows:
        raise SystemExit(f"No rows found in {dataset_path}")

    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in rows[0]]
    if missing:
        raise SystemExit(f"Dataset {dataset_path} is missing required columns: {', '.join(missing)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    x, y, weights = rows_to_tensors(torch, rows)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    # Optional high-quality GBM teacher + distillation
    gbm_model = None
    gbm_soft_targets = None
    if getattr(args, "use_gbm_teacher", False):
        if ensure_lightgbm():
            gbm_result = try_train_gbm_teacher(rows, FEATURE_COLUMNS, LABEL_COLUMN, args, mean, std)
            if gbm_result is not None:
                gbm_model, gbm_soft_targets_np = gbm_result
                gbm_soft_targets = torch.tensor(gbm_soft_targets_np, dtype=torch.float32)
        else:
            print("Continuing without GBM teacher (distillation disabled).")

    validation_fraction = min(0.50, max(0.01, args.validation_fraction))
    val_count = max(1, int(len(indices) * validation_fraction))
    if val_count >= len(indices):
        val_count = max(1, len(indices) // 5)
    val_idx = torch.tensor(indices[:val_count], dtype=torch.long)
    train_idx = torch.tensor(indices[val_count:], dtype=torch.long)

    train_x = x.index_select(0, train_idx)
    train_y = y.index_select(0, train_idx)
    train_w = weights.index_select(0, train_idx)
    val_x = x.index_select(0, val_idx)
    val_y = y.index_select(0, val_idx)
    val_w = weights.index_select(0, val_idx)

    if gbm_soft_targets is not None:
        train_gbm_soft = gbm_soft_targets.index_select(0, train_idx)
        val_gbm_soft = gbm_soft_targets.index_select(0, val_idx)
    else:
        train_gbm_soft = val_gbm_soft = None

    hidden = max(8, args.hidden)
    model = make_model(torch, len(FEATURE_COLUMNS), hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max(1e-7, args.learning_rate), weight_decay=max(0.0, args.weight_decay)
    )
    batch_size = max(16, args.batch_size)
    eps = 1e-6
    writer = make_summary_writer(not args.no_tensorboard, args.log_dir)

    distill_w = float(max(0.0, min(0.95, args.distill_weight)))
    use_distill = train_gbm_soft is not None and distill_w > 0.0

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        order = torch.randperm(train_x.shape[0])
        total_loss = 0.0
        seen = 0
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            bx = train_x.index_select(0, batch_idx)
            by_hard = train_y.index_select(0, batch_idx)
            bw = train_w.index_select(0, batch_idx)

            # Effective target: mix of hard label and GBM soft probability (knowledge distillation)
            if use_distill:
                by_soft = train_gbm_soft.index_select(0, batch_idx)
                by = (1.0 - distill_w) * by_hard + distill_w * by_soft
            else:
                by = by_hard

            optimizer.zero_grad(set_to_none=True)
            pred = model(bx).clamp(eps, 1.0 - eps)
            loss = -torch.mean(bw * (by * torch.log(pred) + (1.0 - by) * torch.log(1.0 - pred)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).clamp(eps, 1.0 - eps)
            # Always evaluate against hard labels for fair comparison
            val_bce_weighted = -torch.mean(val_w * (val_y * torch.log(val_pred) + (1.0 - val_y) * torch.log(1.0 - val_pred)))
            val_mae = torch.mean(torch.abs(val_pred - val_y))
            val_acc = torch.mean(((val_pred >= 0.5) == (val_y >= 0.5)).float())

        train_loss = total_loss / max(1, seen)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val_weighted_bce", float(val_bce_weighted), epoch)
            writer.add_scalar("metrics/val_mae", float(val_mae), epoch)
            writer.add_scalar("metrics/val_acc", float(val_acc), epoch)

        mode = " (distilled)" if use_distill else ""
        print(
            f"epoch={epoch:03d} train_bce={train_loss:.6f} "
            f"val_bce_w={float(val_bce_weighted):.6f} val_mae={float(val_mae):.6f} val_acc={float(val_acc):.4f}{mode}"
        )

        val_scalar = float(val_bce_weighted)
        if val_scalar < best_val - 1e-7:
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

    output_path = resolve_repo_path(args.output)
    metadata_path = resolve_repo_path(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "state_dict": model_cpu.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "hidden": hidden,
        "dataset": str(dataset_path),
        "best_validation_weighted_bce": best_val,
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
    }
    torch.save(artifact, output_path)

    metadata = {
        "model_path": str(output_path),
        "dataset": str(dataset_path),
        "samples": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "hidden": hidden,
        "final_validation_weighted_bce": best_val,
        "final_validation_mae": float(val_mae),
        "final_validation_accuracy": float(val_acc),
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
        "runtime_contract": "C++ feeds 16 raw features; ONNX applies stored mean/std and outputs single sigmoid match score.",
    }
    if gbm_model is not None:
        metadata["gbm_teacher"] = {
            "boosting_rounds": args.gbm_boosting_rounds,
            "learning_rate": args.gbm_learning_rate,
            "max_depth": args.gbm_max_depth,
        }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved neural tracker artifact to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    if use_distill:
        print("  (Model was improved via LightGBM distillation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
