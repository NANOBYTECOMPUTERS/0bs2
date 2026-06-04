#!/usr/bin/env python3
"""Evaluate the neural tracker association model with calibration and grouped ranking metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.neural_tracker.dataset import FEATURE_COLUMNS, LABEL_COLUMN, read_dataset
from training.train_neural_tracker import (
    apply_temperature,
    binary_metrics,
    group_ids_for_rows,
    group_top1_accuracy,
    import_torch,
    make_model,
    resolve_repo_path,
    rows_to_tensors,
)


def _rank_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label >= 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[index]]:
            end += 1
        average_rank = (index + 1 + end) * 0.5
        for rank_index in range(index, end):
            ranks[ordered[rank_index]] = average_rank
        index = end

    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label >= 0.5)
    return (positive_rank_sum - positives * (positives + 1) * 0.5) / max(1, positives * negatives)


def _pr_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label >= 0.5)
    if positives == 0:
        return 0.0

    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0.0
    false_positive = 0.0
    prev_recall = 0.0
    area = 0.0
    for index in ordered:
        if labels[index] >= 0.5:
            true_positive += 1.0
        else:
            false_positive += 1.0
        recall = true_positive / positives
        precision = true_positive / max(1.0, true_positive + false_positive)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def evaluate_model(torch, model_path: Path, rows: list[dict]) -> dict[str, float | int | str]:
    artifact = torch.load(model_path, map_location="cpu")
    feature_columns = artifact["feature_columns"]
    if feature_columns != FEATURE_COLUMNS:
        raise SystemExit(
            f"Feature contract mismatch. Model has {feature_columns}; runtime/training expects {FEATURE_COLUMNS}."
        )

    model = make_model(torch, len(feature_columns), int(artifact["hidden"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    x_raw, y, weights = rows_to_tensors(torch, rows)
    mean = torch.tensor(artifact["feature_mean"], dtype=torch.float32)
    std = torch.tensor(artifact["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    x = (x_raw - mean) / std
    group_ids = torch.tensor(group_ids_for_rows(rows), dtype=torch.long)

    with torch.no_grad():
        raw_pred = model(x).clamp(1e-6, 1.0 - 1e-6)
        pred = apply_temperature(torch, raw_pred, float(artifact.get("calibration_temperature", 1.0))).clamp(1e-6, 1.0 - 1e-6)

    metrics = binary_metrics(torch, pred, y, weights)
    top1, group_count = group_top1_accuracy(pred, y, group_ids)
    labels = y.flatten().cpu().tolist()
    scores = pred.flatten().cpu().tolist()
    metrics.update(
        {
            "samples": len(rows),
            "roc_auc": _rank_auc(labels, scores),
            "pr_auc": _pr_auc(labels, scores),
            "group_top1_accuracy": top1,
            "group_count": group_count,
            "calibration_temperature": float(artifact.get("calibration_temperature", 1.0)),
            "model": str(model_path),
        }
    )
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a neural tracker association model.")
    parser.add_argument("--dataset", default="training/data/neural_tracker_dataset.csv")
    parser.add_argument("--real-world-data", action="append", default=[], help="Optional runtime/log CSV to append.")
    parser.add_argument("--model", default="neural_models/neural_tracker.pt")
    parser.add_argument("--compare-model", default="", help="Optional second model for before/after comparison.")
    parser.add_argument("--output", default="training/models/neural_tracker_eval.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    rows = read_dataset(resolve_repo_path(args.dataset))
    for path in args.real_world_data:
        rows.extend(read_dataset(resolve_repo_path(path)))
    if not rows:
        raise SystemExit("No evaluation rows found.")

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": str(resolve_repo_path(args.dataset)),
        "real_world_data": [str(resolve_repo_path(path)) for path in args.real_world_data],
        "primary": evaluate_model(torch, resolve_repo_path(args.model), rows),
    }
    if args.compare_model:
        comparison = evaluate_model(torch, resolve_repo_path(args.compare_model), rows)
        result["comparison"] = comparison
        result["delta"] = {
            "weighted_bce": result["primary"]["weighted_bce"] - comparison["weighted_bce"],
            "brier": result["primary"]["brier"] - comparison["brier"],
            "group_top1_accuracy": result["primary"]["group_top1_accuracy"] - comparison["group_top1_accuracy"],
        }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved neural tracker evaluation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
