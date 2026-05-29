#!/usr/bin/env python3
"""Dataset helpers for temporal predictor training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = ["x", "y", "w", "h", "vx", "vy", "box_scale_vel", "confidence"]


@dataclass(frozen=True)
class TemporalDataset:
    history: np.ndarray
    future: np.ndarray
    profiles: np.ndarray
    metadata: dict


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[2] / resolved


def _read_metadata(npz) -> dict:
    if "metadata_json" not in npz:
        return {}
    raw = npz["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def load_temporal_npz(path: str | Path) -> TemporalDataset:
    dataset_path = resolve_repo_path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        if "history" not in data or "future" not in data:
            raise ValueError(f"{dataset_path} must contain history and future arrays")
        history = data["history"].astype(np.float32, copy=False)
        future = data["future"].astype(np.float32, copy=False)
        profiles = data["profiles"] if "profiles" in data else np.array([], dtype="<U1")
        metadata = _read_metadata(data)

    if history.ndim != 3 or history.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"history must have shape [samples, history_length, {len(FEATURE_COLUMNS)}], "
            f"got {history.shape}"
        )
    if future.ndim != 3 or future.shape[-1] != 2:
        raise ValueError(f"future must have shape [samples, prediction_horizon, 2], got {future.shape}")
    if history.shape[0] != future.shape[0]:
        raise ValueError("history and future sample counts must match")
    if history.shape[0] < 2:
        raise ValueError("temporal dataset must contain at least two samples")

    metadata.setdefault("feature_columns", FEATURE_COLUMNS)
    metadata.setdefault("history_length", int(history.shape[1]))
    metadata.setdefault("prediction_horizon", int(future.shape[1]))
    metadata.setdefault("samples", int(history.shape[0]))
    return TemporalDataset(history=history, future=future, profiles=profiles, metadata=metadata)
