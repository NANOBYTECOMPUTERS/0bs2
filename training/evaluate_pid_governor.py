#!/usr/bin/env python3
"""Evaluate a trained PID governor model against a dataset CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pid_governor.dataset import FEATURE_COLUMNS, LABEL_COLUMNS, read_dataset
from training.train_pid_governor import import_torch, make_model


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PID governor model fit.")
    parser.add_argument("--dataset", default="training/data/pid_governor_dataset.csv")
    parser.add_argument("--model", default="training/models/pid_governor.pt")
    parser.add_argument("--predictions", default="", help="Optional CSV path for predictions.")
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
            "Regenerate it with generate_pid_dataset.py before evaluating."
        )

    artifact = torch.load(resolve_repo_path(args.model), map_location="cpu")
    model = make_model(torch, len(FEATURE_COLUMNS), len(LABEL_COLUMNS), int(artifact["hidden"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    x = torch.tensor([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=torch.float32)
    y = torch.tensor([[float(row[column]) for column in LABEL_COLUMNS] for row in rows], dtype=torch.float32)
    mean = torch.tensor(artifact["feature_mean"], dtype=torch.float32)
    std = torch.tensor(artifact["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    x = (x - mean) / std

    with torch.no_grad():
        pred = model(x)
        mse = torch.mean((pred - y) ** 2, dim=0)
        mae = torch.mean(torch.abs(pred - y), dim=0)

    print("Evaluation:")
    for index, label in enumerate(LABEL_COLUMNS):
        print(f"  {label}: mse={float(mse[index]):.6f} mae={float(mae[index]):.6f}")
    print(f"  overall_mse={float(torch.mean(mse)):.6f} overall_mae={float(torch.mean(mae)):.6f}")

    if args.predictions:
        output = resolve_repo_path(args.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["profile", "episode", "step"]
            fieldnames += [f"target_{label}" for label in LABEL_COLUMNS]
            fieldnames += [f"pred_{label}" for label in LABEL_COLUMNS]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, target_values, pred_values in zip(rows, y.tolist(), pred.tolist()):
                out = {
                    "profile": row["profile"],
                    "episode": row["episode"],
                    "step": row["step"],
                }
                for label, target_value, pred_value in zip(LABEL_COLUMNS, target_values, pred_values):
                    out[f"target_{label}"] = target_value
                    out[f"pred_{label}"] = pred_value
                writer.writerow(out)
        print(f"Wrote predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
