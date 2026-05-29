#!/usr/bin/env python3
"""Dataset helpers for the optional neural tracker association model."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


FEATURE_COLUMNS = [
    "distance_norm",
    "iou",
    "size_log_ratio",
    "detection_confidence",
    "track_confidence",
    "heading_alignment",
    "track_missed_norm",
    "track_hits_norm",
    "is_locked",
    "class_compatible",
    "dt",
    "speed_norm",
    "target_size_norm",
    "pivot_offset_x_norm",
    "pivot_offset_y_norm",
    "relaxed_gate",
]

LABEL_COLUMN = "label_match"

OUTPUT_COLUMNS = ["source", *FEATURE_COLUMNS, LABEL_COLUMN, "sample_weight"]


@dataclass(frozen=True)
class NeuralTrackerDatasetConfig:
    samples: int = 20000
    seed: int = 1337
    positive_fraction: float = 0.50
    hard_negative_fraction: float = 0.70


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return min(hi, max(lo, value))


def _float_or_default(value: str | float | int | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_from_runtime_row(row: dict[str, str]) -> float:
    if LABEL_COLUMN in row and row[LABEL_COLUMN] not in {None, ""}:
        return 1.0 if _float_or_default(row[LABEL_COLUMN], 0.0) >= 0.5 else 0.0
    if "chosen" in row and row["chosen"] not in {None, ""}:
        return 1.0 if _float_or_default(row["chosen"], 0.0) >= 0.5 else 0.0
    if "accepted" in row and row["accepted"] not in {None, ""}:
        return 1.0 if _float_or_default(row["accepted"], 0.0) >= 0.5 else 0.0
    return 0.0


def normalize_row(row: dict[str, str | float | int], source: str = "log") -> dict[str, float | str]:
    normalized: dict[str, float | str] = {"source": str(row.get("source") or source)}
    for column in FEATURE_COLUMNS:
        normalized[column] = _float_or_default(row.get(column), 0.0)
    normalized["distance_norm"] = _clamp(float(normalized["distance_norm"]), 0.0, 3.0)
    normalized["iou"] = _clamp(float(normalized["iou"]), 0.0, 1.0)
    normalized["size_log_ratio"] = _clamp(float(normalized["size_log_ratio"]), -3.0, 3.0)
    normalized["detection_confidence"] = _clamp(float(normalized["detection_confidence"]), 0.0, 1.0)
    normalized["track_confidence"] = _clamp(float(normalized["track_confidence"]), 0.0, 1.0)
    normalized["heading_alignment"] = _clamp(float(normalized["heading_alignment"]), -1.0, 1.0)
    normalized["track_missed_norm"] = _clamp(float(normalized["track_missed_norm"]), 0.0, 1.0)
    normalized["track_hits_norm"] = _clamp(float(normalized["track_hits_norm"]), 0.0, 1.0)
    normalized["is_locked"] = _clamp(float(normalized["is_locked"]), 0.0, 1.0)
    normalized["class_compatible"] = _clamp(float(normalized["class_compatible"]), 0.0, 1.0)
    normalized["dt"] = _clamp(float(normalized["dt"]), 0.0, 0.25)
    normalized["speed_norm"] = _clamp(float(normalized["speed_norm"]), 0.0, 2.0)
    normalized["target_size_norm"] = _clamp(float(normalized["target_size_norm"]), 0.0, 1.0)
    normalized["pivot_offset_x_norm"] = _clamp(float(normalized["pivot_offset_x_norm"]), -2.0, 2.0)
    normalized["pivot_offset_y_norm"] = _clamp(float(normalized["pivot_offset_y_norm"]), -2.0, 2.0)
    normalized["relaxed_gate"] = _clamp(float(normalized["relaxed_gate"]), 0.0, 1.0)
    normalized[LABEL_COLUMN] = _label_from_runtime_row(row)
    normalized["sample_weight"] = max(0.05, _float_or_default(row.get("sample_weight"), 1.0))
    return normalized


def _positive_sample(rng: random.Random) -> dict[str, float | str]:
    locked = 1.0 if rng.random() < 0.35 else 0.0
    relaxed = 1.0 if locked > 0.5 and rng.random() < 0.25 else 0.0
    distance = rng.betavariate(1.3, 5.5) * (0.85 if relaxed < 0.5 else 1.35)
    iou = 1.0 - rng.betavariate(1.2, 4.8) * 0.50
    return {
        "source": "synthetic_positive",
        "distance_norm": _clamp(distance, 0.0, 3.0),
        "iou": _clamp(iou, 0.0, 1.0),
        "size_log_ratio": _clamp(rng.gauss(0.0, 0.28), -3.0, 3.0),
        "detection_confidence": rng.uniform(0.62, 1.0),
        "track_confidence": rng.uniform(0.56, 1.0),
        "heading_alignment": _clamp(rng.gauss(0.70, 0.32), -1.0, 1.0),
        "track_missed_norm": rng.betavariate(1.0, 7.0),
        "track_hits_norm": rng.betavariate(4.5, 1.6),
        "is_locked": locked,
        "class_compatible": 1.0,
        "dt": rng.uniform(0.001, 0.055),
        "speed_norm": _clamp(rng.betavariate(1.8, 5.0) * 1.2, 0.0, 2.0),
        "target_size_norm": rng.uniform(0.006, 0.18),
        "pivot_offset_x_norm": _clamp(rng.gauss(0.0, 0.030), -2.0, 2.0),
        "pivot_offset_y_norm": _clamp(rng.gauss(0.0, 0.035), -2.0, 2.0),
        "relaxed_gate": relaxed,
        LABEL_COLUMN: 1.0,
        "sample_weight": 1.0,
    }


def _negative_sample(rng: random.Random, hard: bool) -> dict[str, float | str]:
    mode = rng.choice(["far", "overlap", "size", "heading", "class", "pivot", "low_conf"])
    if hard:
        distance = rng.uniform(0.25, 1.35)
        iou = rng.uniform(0.10, 0.65)
        size_log = rng.uniform(-1.2, 1.2)
        heading = rng.uniform(-0.35, 0.85)
        class_compatible = 1.0
        det_conf = rng.uniform(0.45, 1.0)
        track_conf = rng.uniform(0.40, 1.0)
    else:
        distance = rng.uniform(1.10, 3.0)
        iou = rng.uniform(0.0, 0.28)
        size_log = rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 3.0)
        heading = rng.uniform(-1.0, 0.25)
        class_compatible = 1.0 if rng.random() < 0.70 else 0.0
        det_conf = rng.uniform(0.10, 0.90)
        track_conf = rng.uniform(0.10, 0.90)

    if mode == "far":
        distance = rng.uniform(1.1 if hard else 1.8, 3.0)
    elif mode == "overlap":
        iou = rng.uniform(0.0, 0.20 if hard else 0.10)
    elif mode == "size":
        size_log = rng.choice([-1.0, 1.0]) * rng.uniform(1.0 if hard else 1.6, 3.0)
    elif mode == "heading":
        heading = rng.uniform(-1.0, -0.10)
    elif mode == "class":
        class_compatible = 0.0
    elif mode == "pivot":
        distance = min(distance, rng.uniform(0.10, 0.85))
    elif mode == "low_conf":
        det_conf = rng.uniform(0.0, 0.35)

    pivot_sigma = 0.05 if mode != "pivot" else rng.uniform(0.20, 0.85)
    return {
        "source": "synthetic_hard_negative" if hard else "synthetic_negative",
        "distance_norm": _clamp(distance, 0.0, 3.0),
        "iou": _clamp(iou, 0.0, 1.0),
        "size_log_ratio": _clamp(size_log, -3.0, 3.0),
        "detection_confidence": _clamp(det_conf, 0.0, 1.0),
        "track_confidence": _clamp(track_conf, 0.0, 1.0),
        "heading_alignment": _clamp(heading, -1.0, 1.0),
        "track_missed_norm": rng.betavariate(1.4, 3.0),
        "track_hits_norm": rng.betavariate(1.5, 3.2),
        "is_locked": 1.0 if rng.random() < 0.18 else 0.0,
        "class_compatible": class_compatible,
        "dt": rng.uniform(0.001, 0.080),
        "speed_norm": _clamp(rng.betavariate(2.0, 2.0) * 1.8, 0.0, 2.0),
        "target_size_norm": rng.uniform(0.006, 0.20),
        "pivot_offset_x_norm": _clamp(rng.gauss(0.0, pivot_sigma), -2.0, 2.0),
        "pivot_offset_y_norm": _clamp(rng.gauss(0.0, pivot_sigma), -2.0, 2.0),
        "relaxed_gate": 1.0 if rng.random() < 0.12 else 0.0,
        LABEL_COLUMN: 0.0,
        "sample_weight": 1.25 if hard else 1.0,
    }


def generate_synthetic_dataset(cfg: NeuralTrackerDatasetConfig) -> list[dict[str, float | str]]:
    rng = random.Random(cfg.seed)
    samples = max(2, int(cfg.samples))
    positive_target = int(samples * _clamp(cfg.positive_fraction, 0.05, 0.95))
    rows: list[dict[str, float | str]] = []
    for index in range(samples):
        if index < positive_target:
            rows.append(_positive_sample(rng))
        else:
            hard = rng.random() < _clamp(cfg.hard_negative_fraction, 0.0, 1.0)
            rows.append(_negative_sample(rng, hard))
    rng.shuffle(rows)
    return rows


def write_dataset(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            normalized = normalize_row(row, source=str(row.get("source") or "dataset"))
            writer.writerow({column: normalized.get(column, "") for column in OUTPUT_COLUMNS})


def read_dataset(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [normalize_row(row, source=path.stem) for row in reader]
    return rows


def merge_logs(paths: Iterable[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(read_dataset(path))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic neural tracker association dataset.")
    parser.add_argument("--output", default="training/data/neural_tracker_dataset.csv")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--positive-fraction", type=float, default=0.50)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.70)
    parser.add_argument(
        "--merge-log",
        action="append",
        default=[],
        help="Optional neural_tracker_association.csv log to append to the synthetic dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = NeuralTrackerDatasetConfig(
        samples=args.samples,
        seed=args.seed,
        positive_fraction=args.positive_fraction,
        hard_negative_fraction=args.hard_negative_fraction,
    )
    rows = generate_synthetic_dataset(cfg)
    rows.extend(merge_logs(resolve_repo_path(path) for path in args.merge_log))
    output = resolve_repo_path(args.output)
    write_dataset(output, rows)
    positives = sum(1 for row in rows if float(row[LABEL_COLUMN]) >= 0.5)
    print(f"Wrote {len(rows)} neural tracker samples to {output}")
    print(f"Positive samples: {positives}; negative samples: {len(rows) - positives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
