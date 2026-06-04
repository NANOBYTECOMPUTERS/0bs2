#!/usr/bin/env python3
"""Train the optional neural tracker association MLP."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from collections import defaultdict
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

from training.neural_tracker.dataset import FEATURE_COLUMNS, GROUP_COLUMN, LABEL_COLUMN, read_dataset


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


def try_train_gbm_teacher(train_rows, predict_rows, feature_columns, label_column, args, mean, std):
    """Train a LightGBM teacher on train rows only, then predict soft labels for all rows."""
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        print("LightGBM not installed. Falling back to pure neural training. "
              "Install with: pip install lightgbm")
        return None

    print(f"Training LightGBM teacher ({args.gbm_boosting_rounds} rounds)...")
    train_x = np.array([[float(row[col]) for col in feature_columns] for row in train_rows], dtype=np.float32)
    train_x = (train_x - mean.numpy()) / std.numpy()
    train_y = np.array([float(row[label_column]) for row in train_rows], dtype=np.float32)
    train_w = np.array([float(row.get("sample_weight", 1.0)) for row in train_rows], dtype=np.float32).clip(0.05, None)

    predict_x = np.array([[float(row[col]) for col in feature_columns] for row in predict_rows], dtype=np.float32)
    predict_x = (predict_x - mean.numpy()) / std.numpy()

    train_data = lgb.Dataset(train_x, label=train_y, weight=train_w)

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

    soft_probs = gbm.predict(predict_x, raw_score=False)
    print(f"  GBM teacher trained. Example soft prob range: [{soft_probs.min():.3f}, {soft_probs.max():.3f}]")

    return gbm, soft_probs.astype(np.float32)


def group_key_for_row(row: dict, row_index: int) -> str:
    group_value = str(row.get(GROUP_COLUMN, "")).strip()
    if group_value:
        return group_value
    session = str(row.get("session_id", "")).strip()
    frame = str(row.get("frame_id", "")).strip()
    track = str(row.get("track_id", "")).strip()
    if session or frame or track:
        return f"session={session}|frame={frame}|track={track}"
    return f"row={row_index}"


def split_indices_by_group(rows, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key_for_row(row, index)].append(index)

    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    validation_fraction = min(0.50, max(0.01, validation_fraction))
    target_val_count = max(1, int(len(rows) * validation_fraction))
    val_indices: list[int] = []
    train_indices: list[int] = []

    for key in group_keys:
        target = val_indices if len(val_indices) < target_val_count else train_indices
        target.extend(groups[key])

    if not train_indices:
        train_indices, val_indices = val_indices[:-1], val_indices[-1:]
    if not val_indices:
        val_indices, train_indices = train_indices[:1], train_indices[1:]
    return train_indices, val_indices


def group_ids_for_rows(rows) -> list[int]:
    id_by_key: dict[str, int] = {}
    ids: list[int] = []
    for index, row in enumerate(rows):
        key = group_key_for_row(row, index)
        if key not in id_by_key:
            id_by_key[key] = len(id_by_key)
        ids.append(id_by_key[key])
    return ids


def apply_temperature(torch, probs, temperature: float):
    if hasattr(temperature, "detach"):
        temp = torch.clamp(temperature, 0.05, 20.0)
    else:
        temp = torch.tensor(max(0.05, float(temperature)), dtype=probs.dtype, device=probs.device)
    logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
    return torch.sigmoid(logits / temp)


def expected_calibration_error(torch, preds, labels, bins: int = 10) -> float:
    pred = preds.detach().flatten()
    label = labels.detach().flatten()
    total = max(1, int(pred.numel()))
    ece = torch.tensor(0.0, dtype=pred.dtype)
    for bin_index in range(max(1, bins)):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (pred >= lower) & (pred <= upper)
        else:
            mask = (pred >= lower) & (pred < upper)
        if torch.any(mask):
            ece = ece + torch.abs(pred[mask].mean() - label[mask].mean()) * (mask.float().sum() / total)
    return float(ece.cpu())


def binary_metrics(torch, preds, labels, weights, eps: float = 1e-6) -> dict[str, float]:
    pred = preds.clamp(eps, 1.0 - eps)
    bce = -torch.mean(weights * (labels * torch.log(pred) + (1.0 - labels) * torch.log(1.0 - pred)))
    mae = torch.mean(torch.abs(pred - labels))
    acc = torch.mean(((pred >= 0.5) == (labels >= 0.5)).float())
    brier = torch.mean((pred - labels) * (pred - labels))
    return {
        "weighted_bce": float(bce.cpu()),
        "mae": float(mae.cpu()),
        "accuracy": float(acc.cpu()),
        "brier": float(brier.cpu()),
        "ece": expected_calibration_error(torch, pred, labels),
    }


def pairwise_ranking_loss(torch, preds, labels, group_ids, margin: float):
    losses = []
    flat_pred = preds.flatten()
    flat_label = labels.flatten()
    for group_id in torch.unique(group_ids).tolist():
        mask = group_ids == group_id
        group_pred = flat_pred[mask]
        group_label = flat_label[mask]
        positives = group_pred[group_label >= 0.5]
        negatives = group_pred[group_label < 0.5]
        if positives.numel() > 0 and negatives.numel() > 0:
            losses.append(torch.relu(float(margin) - (positives[:, None] - negatives[None, :])).mean())
    if not losses:
        return flat_pred.sum() * 0.0
    return torch.stack(losses).mean()


def group_top1_accuracy(preds, labels, group_ids) -> tuple[float, int]:
    pred_values = preds.detach().flatten().cpu().tolist()
    label_values = labels.detach().flatten().cpu().tolist()
    group_values = group_ids.detach().flatten().cpu().tolist()
    groups: dict[int, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_values):
        groups[int(group_id)].append(index)

    total = 0
    correct = 0
    for indices in groups.values():
        if not any(label_values[i] >= 0.5 for i in indices):
            continue
        if not any(label_values[i] < 0.5 for i in indices):
            continue
        best_index = max(indices, key=lambda i: pred_values[i])
        correct += 1 if label_values[best_index] >= 0.5 else 0
        total += 1
    return (correct / total if total else 0.0), total


def fit_temperature(torch, model, val_x, val_y, val_w) -> float:
    model.eval()
    with torch.no_grad():
        base_probs = model(val_x).clamp(1e-6, 1.0 - 1e-6)

    log_temp = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temp).clamp(0.05, 20.0)
        calibrated = apply_temperature(torch, base_probs, temperature).clamp(1e-6, 1.0 - 1e-6)
        loss = -torch.mean(val_w * (val_y * torch.log(calibrated) + (1.0 - val_y) * torch.log(1.0 - calibrated)))
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return 1.0
    return float(torch.exp(log_temp).detach().clamp(0.05, 20.0).cpu())


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
    parser.add_argument(
        "--real-world-data",
        action="append",
        default=[],
        help="Optional neural tracker CSV/log dataset to append to the primary dataset.",
    )
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
    parser.add_argument("--ranking-loss-weight", type=float, default=0.15,
                        help="Pairwise within-group ranking loss weight. Use 0.0 to disable.")
    parser.add_argument("--ranking-margin", type=float, default=0.15,
                        help="Desired score margin between positive and negative candidates in the same group.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Disable validation-set temperature calibration in the saved/exported model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    dataset_path = resolve_repo_path(args.dataset)
    rows = read_dataset(dataset_path)
    real_world_paths = [resolve_repo_path(path) for path in args.real_world_data]
    for real_world_path in real_world_paths:
        real_rows = read_dataset(real_world_path)
        rows.extend(real_rows)
        print(f"Loaded {len(real_rows)} optional real-world/log rows from {real_world_path}")
    if not rows:
        raise SystemExit(f"No rows found in {dataset_path}")

    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in rows[0]]
    if missing:
        raise SystemExit(f"Dataset {dataset_path} is missing required columns: {', '.join(missing)}")

    torch.manual_seed(args.seed)

    train_indices, val_indices = split_indices_by_group(rows, args.validation_fraction, args.seed)
    val_idx = torch.tensor(val_indices, dtype=torch.long)
    train_idx = torch.tensor(train_indices, dtype=torch.long)

    x_raw, y, weights = rows_to_tensors(torch, rows)
    mean = x_raw.index_select(0, train_idx).mean(dim=0)
    std = x_raw.index_select(0, train_idx).std(dim=0).clamp_min(1e-6)
    x = (x_raw - mean) / std
    group_ids = torch.tensor(group_ids_for_rows(rows), dtype=torch.long)

    train_x = x.index_select(0, train_idx)
    train_y = y.index_select(0, train_idx)
    train_w = weights.index_select(0, train_idx)
    train_g = group_ids.index_select(0, train_idx)
    val_x = x.index_select(0, val_idx)
    val_y = y.index_select(0, val_idx)
    val_w = weights.index_select(0, val_idx)
    val_g = group_ids.index_select(0, val_idx)

    # Optional high-quality GBM teacher + distillation. Train only on train rows to avoid validation leakage.
    gbm_model = None
    gbm_soft_targets = None
    if getattr(args, "use_gbm_teacher", False):
        if ensure_lightgbm():
            train_rows = [rows[index] for index in train_indices]
            gbm_result = try_train_gbm_teacher(train_rows, rows, FEATURE_COLUMNS, LABEL_COLUMN, args, mean, std)
            if gbm_result is not None:
                gbm_model, gbm_soft_targets_np = gbm_result
                gbm_soft_targets = torch.tensor(gbm_soft_targets_np, dtype=torch.float32)
        else:
            print("Continuing without GBM teacher (distillation disabled).")

    if gbm_soft_targets is not None:
        train_gbm_soft = gbm_soft_targets.index_select(0, train_idx)
    else:
        train_gbm_soft = None

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
    ranking_weight = float(max(0.0, args.ranking_loss_weight))
    ranking_margin = float(max(0.0, args.ranking_margin))

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
            bg = train_g.index_select(0, batch_idx)

            # Effective target: mix of hard label and GBM soft probability (knowledge distillation)
            if use_distill:
                by_soft = train_gbm_soft.index_select(0, batch_idx)
                by = (1.0 - distill_w) * by_hard + distill_w * by_soft
            else:
                by = by_hard

            optimizer.zero_grad(set_to_none=True)
            pred = model(bx).clamp(eps, 1.0 - eps)
            bce_loss = -torch.mean(bw * (by * torch.log(pred) + (1.0 - by) * torch.log(1.0 - pred)))
            if ranking_weight > 0.0:
                rank_loss = pairwise_ranking_loss(torch, pred, by_hard, bg, ranking_margin)
                loss = bce_loss + ranking_weight * rank_loss
            else:
                loss = bce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).clamp(eps, 1.0 - eps)
            val_metrics = binary_metrics(torch, val_pred, val_y, val_w, eps=eps)
            val_top1, val_group_count = group_top1_accuracy(val_pred, val_y, val_g)

        train_loss = total_loss / max(1, seen)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val_weighted_bce", val_metrics["weighted_bce"], epoch)
            writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch)
            writer.add_scalar("metrics/val_acc", val_metrics["accuracy"], epoch)
            writer.add_scalar("metrics/val_brier", val_metrics["brier"], epoch)
            writer.add_scalar("metrics/val_ece", val_metrics["ece"], epoch)
            writer.add_scalar("metrics/val_group_top1", val_top1, epoch)

        mode = " (distilled)" if use_distill else ""
        print(
            f"epoch={epoch:03d} train_bce={train_loss:.6f} "
            f"val_bce_w={val_metrics['weighted_bce']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_brier={val_metrics['brier']:.6f} "
            f"val_ece={val_metrics['ece']:.6f} val_top1={val_top1:.4f}/{val_group_count}{mode}"
        )

        val_scalar = val_metrics["weighted_bce"]
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
    calibration_temperature = 1.0
    if not args.no_calibration:
        calibration_temperature = fit_temperature(torch, model_cpu, val_x, val_y, val_w)

    with torch.no_grad():
        final_raw_pred = model_cpu(val_x).clamp(eps, 1.0 - eps)
        final_pred = apply_temperature(torch, final_raw_pred, calibration_temperature)
        final_metrics = binary_metrics(torch, final_pred, val_y, val_w, eps=eps)
        final_top1, final_group_count = group_top1_accuracy(final_pred, val_y, val_g)

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
        "calibration_temperature": calibration_temperature,
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
        "ranking_loss_weight": ranking_weight,
        "ranking_margin": ranking_margin,
    }
    torch.save(artifact, output_path)

    metadata = {
        "model_path": str(output_path),
        "dataset": str(dataset_path),
        "samples": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "hidden": hidden,
        "final_validation_weighted_bce": final_metrics["weighted_bce"],
        "final_validation_mae": final_metrics["mae"],
        "final_validation_accuracy": final_metrics["accuracy"],
        "final_validation_brier": final_metrics["brier"],
        "final_validation_ece": final_metrics["ece"],
        "final_validation_group_top1_accuracy": final_top1,
        "final_validation_group_count": final_group_count,
        "calibration_temperature": calibration_temperature,
        "distilled_from_gbm": bool(use_distill),
        "distill_weight": distill_w if use_distill else 0.0,
        "split_strategy": "group_id/session-frame-track grouped validation",
        "ranking_loss_weight": ranking_weight,
        "ranking_margin": ranking_margin,
        "real_world_datasets": [str(path) for path in real_world_paths],
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
