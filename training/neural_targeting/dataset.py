#!/usr/bin/env python3
"""Dataset helpers for neural targeting head training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_FEATURE_COLUMNS = [
    "center_x",
    "center_y",
    "width",
    "height",
    "vx",
    "vy",
    "box_scale_vel",
    "confidence",
    "refinement_x",
    "refinement_y",
]
LABEL_COLUMNS = ["refinement_offset_x", "refinement_offset_y", "confidence"]


@dataclass(frozen=True)
class NeuralTargetingDataset:
    features: np.ndarray
    refinement: np.ndarray
    confidence: np.ndarray
    metadata: dict


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[2] / resolved


def feature_columns_for_horizon(prediction_horizon: int) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
    for i in range(max(1, int(prediction_horizon))):
        columns.append(f"future_dx_{i}")
        columns.append(f"future_dy_{i}")
    return columns


def _read_metadata(npz) -> dict:
    if "metadata_json" not in npz:
        return {}
    raw = npz["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def load_neural_targeting_npz(path: str | Path) -> NeuralTargetingDataset:
    dataset_path = resolve_repo_path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        for key in ("features", "refinement", "confidence"):
            if key not in data:
                raise ValueError(f"{dataset_path} must contain {key}")

        features = data["features"].astype(np.float32, copy=False)
        refinement = data["refinement"].astype(np.float32, copy=False)
        confidence = data["confidence"].astype(np.float32, copy=False)
        metadata = _read_metadata(data)

    if features.ndim != 2:
        raise ValueError(f"features must have shape [samples, feature_dim], got {features.shape}")
    if refinement.ndim != 2 or refinement.shape[-1] != 2:
        raise ValueError(f"refinement must have shape [samples, 2], got {refinement.shape}")
    if confidence.ndim == 1:
        confidence = confidence.reshape(-1, 1)
    if confidence.ndim != 2 or confidence.shape[-1] != 1:
        raise ValueError(f"confidence must have shape [samples, 1], got {confidence.shape}")
    if features.shape[0] != refinement.shape[0] or features.shape[0] != confidence.shape[0]:
        raise ValueError("features, refinement, and confidence sample counts must match")
    if features.shape[0] < 2:
        raise ValueError("neural targeting dataset must contain at least two samples")

    prediction_horizon = int(metadata.get("prediction_horizon", max(1, (features.shape[1] - len(BASE_FEATURE_COLUMNS)) // 2)))
    expected_feature_count = len(BASE_FEATURE_COLUMNS) + prediction_horizon * 2
    if features.shape[1] != expected_feature_count:
        raise ValueError(
            f"features must have {expected_feature_count} columns for prediction_horizon={prediction_horizon}, "
            f"got {features.shape[1]}"
        )

    metadata.setdefault("feature_columns", feature_columns_for_horizon(prediction_horizon))
    metadata.setdefault("label_columns", LABEL_COLUMNS)
    metadata.setdefault("prediction_horizon", prediction_horizon)
    metadata.setdefault("input_feature_count", int(features.shape[1]))
    metadata.setdefault("samples", int(features.shape[0]))
    return NeuralTargetingDataset(
        features=features,
        refinement=refinement,
        confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32, copy=False),
        metadata=metadata,
    )

