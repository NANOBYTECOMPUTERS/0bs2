#!/usr/bin/env python3
"""Train a small MLP that outputs PID gain/speed scales."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PID governor MLP.")
    parser.add_argument("--dataset", default="training/data/pid_governor_dataset.csv")
    parser.add_argument("--output", default="training/models/pid_governor.pt")
    parser.add_argument("--metadata", default="training/models/pid_governor.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
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

    x, y = rows_to_tensors(torch, rows)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    val_count = max(1, int(len(indices) * max(0.01, min(0.50, args.validation_fraction))))
    val_idx = torch.tensor(indices[:val_count], dtype=torch.long)
    train_idx = torch.tensor(indices[val_count:], dtype=torch.long)
    train_x = x.index_select(0, train_idx)
    train_y = y.index_select(0, train_idx)
    val_x = x.index_select(0, val_idx)
    val_y = y.index_select(0, val_idx)

    model = make_model(torch, len(FEATURE_COLUMNS), len(LABEL_COLUMNS), max(8, args.hidden))
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(1e-6, args.learning_rate), weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    batch_size = max(8, args.batch_size)

    for epoch in range(1, max(1, args.epochs) + 1):
        model.train()
        order = torch.randperm(train_x.shape[0])
        total_loss = 0.0
        seen = 0
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            bx = train_x.index_select(0, batch_idx)
            by = train_y.index_select(0, batch_idx)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * bx.shape[0]
            seen += bx.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).item())
            val_mae = float(torch.mean(torch.abs(val_pred - val_y)).item())

        train_loss = total_loss / max(1, seen)
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} val_mae={val_mae:.6f}")

    output_path = resolve_repo_path(args.output)
    metadata_path = resolve_repo_path(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "state_dict": model.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "hidden": max(8, args.hidden),
        "dataset": str(dataset_path),
    }
    torch.save(artifact, output_path)

    metadata = {
        "model_path": str(output_path),
        "dataset": str(dataset_path),
        "samples": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "hidden": max(8, args.hidden),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "final_validation_mse": val_loss,
        "final_validation_mae": val_mae,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model artifact to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
