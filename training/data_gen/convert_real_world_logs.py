#!/usr/bin/env python3
"""Convert optional runtime real-world logs into temporal and targeting datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data_gen.generate_targeting_data import _clamp_vector, _cpp_scaled_features, _ideal_lead, near_lock_damping
from training.neural_targeting.dataset import LABEL_COLUMNS, feature_columns_for_horizon, load_neural_targeting_npz
from training.temporal_predictor.dataset import FEATURE_COLUMNS, load_temporal_npz, resolve_repo_path


STATE_COLUMNS = [
    "timestamp_sec",
    "track_id",
    "smooth_x",
    "smooth_y",
    "width",
    "height",
    "vx",
    "vy",
    "box_scale_vel",
    "confidence",
    "mouse_dx",
    "mouse_dy",
    "aim_offset_x",
    "aim_offset_y",
    "detection_resolution",
    "observed_this_frame",
]


def discover_inputs(inputs: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = resolve_repo_path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.npz")))
        elif any(ch in str(path) for ch in "*?[]"):
            paths.extend(sorted(path.parent.glob(path.name)))
        else:
            paths.append(path)
    return [path for path in paths if path.exists()]


def load_real_world_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    log_path = resolve_repo_path(path)
    with np.load(log_path, allow_pickle=False) as data:
        if "history" not in data or "state" not in data:
            raise ValueError(f"{log_path} must contain history and state arrays")
        history = data["history"].astype(np.float32, copy=False)
        state = data["state"].astype(np.float32, copy=False)

    if history.ndim != 3 or history.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(f"{log_path}: history must have shape [samples, history_length, 8], got {history.shape}")
    if state.ndim != 2 or state.shape[-1] != len(STATE_COLUMNS):
        raise ValueError(f"{log_path}: state must have shape [samples, {len(STATE_COLUMNS)}], got {state.shape}")
    if history.shape[0] != state.shape[0]:
        raise ValueError(f"{log_path}: history/state sample counts differ")
    return history, state


def _profile_array(count: int) -> np.ndarray:
    return np.full((count,), "real_world", dtype="<U24")


def _mix_temporal(real_history, real_future, synthetic_path, mix_ratio: float, seed: int):
    if synthetic_path is None:
        return real_history, real_future, _profile_array(real_history.shape[0])

    synthetic = load_temporal_npz(synthetic_path)
    if synthetic.history.shape[1:] != real_history.shape[1:] or synthetic.future.shape[1:] != real_future.shape[1:]:
        raise ValueError("synthetic temporal dataset shape does not match real-world conversion output")

    ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    if ratio >= 1.0:
        return real_history, real_future, _profile_array(real_history.shape[0])
    if ratio <= 0.0:
        return synthetic.history, synthetic.future, synthetic.profiles

    desired_synth = int(round(real_history.shape[0] * (1.0 - ratio) / max(1e-6, ratio)))
    desired_synth = max(0, min(desired_synth, synthetic.history.shape[0]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(synthetic.history.shape[0], size=desired_synth, replace=False) if desired_synth else np.array([], dtype=np.int64)
    history = np.concatenate([real_history, synthetic.history[idx]], axis=0)
    future = np.concatenate([real_future, synthetic.future[idx]], axis=0)
    profiles = np.concatenate([_profile_array(real_history.shape[0]), synthetic.profiles[idx]], axis=0)
    return history, future, profiles


def _mix_targeting(real_features, real_refinement, real_confidence, synthetic_path, mix_ratio: float, seed: int):
    if synthetic_path is None:
        return real_features, real_refinement, real_confidence

    synthetic = load_neural_targeting_npz(synthetic_path)
    if synthetic.features.shape[1:] != real_features.shape[1:]:
        raise ValueError("synthetic targeting dataset feature shape does not match real-world conversion output")

    ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    if ratio >= 1.0:
        return real_features, real_refinement, real_confidence
    if ratio <= 0.0:
        return synthetic.features, synthetic.refinement, synthetic.confidence

    desired_synth = int(round(real_features.shape[0] * (1.0 - ratio) / max(1e-6, ratio)))
    desired_synth = max(0, min(desired_synth, synthetic.features.shape[0]))
    rng = np.random.default_rng(seed + 17)
    idx = rng.choice(synthetic.features.shape[0], size=desired_synth, replace=False) if desired_synth else np.array([], dtype=np.int64)
    features = np.concatenate([real_features, synthetic.features[idx]], axis=0)
    refinement = np.concatenate([real_refinement, synthetic.refinement[idx]], axis=0)
    confidence = np.concatenate([real_confidence, synthetic.confidence[idx]], axis=0)
    return features, refinement, confidence


def build_real_world_datasets(logs: list[Path], history_length: int, prediction_horizon: int):
    history_length = max(2, int(history_length))
    prediction_horizon = max(1, int(prediction_horizon))

    temporal_history: list[np.ndarray] = []
    temporal_future: list[np.ndarray] = []
    targeting_features: list[np.ndarray] = []
    targeting_refinement: list[np.ndarray] = []
    targeting_confidence: list[float] = []

    for path in logs:
        history, state = load_real_world_log(path)
        track_ids = state[:, 1].astype(np.int64)
        for track_id in sorted(set(track_ids.tolist())):
            idx = np.flatnonzero(track_ids == track_id)
            if idx.size <= prediction_horizon:
                continue
            idx = idx[np.argsort(state[idx, 0])]
            for pos in range(0, idx.size - prediction_horizon):
                row = idx[pos]
                future_rows = idx[pos + 1 : pos + 1 + prediction_horizon]
                current = state[row, 2:4].astype(np.float32)
                future = state[future_rows, 2:4].astype(np.float32)
                current_history = history[row].astype(np.float32)
                if current_history.shape[0] != history_length:
                    current_history = current_history[-history_length:]
                    if current_history.shape[0] < history_length:
                        pad = np.repeat(current_history[:1], history_length - current_history.shape[0], axis=0)
                        current_history = np.concatenate([pad, current_history], axis=0)

                temporal_history.append(current_history)
                temporal_future.append(future)

                confidence = float(np.clip(state[row, 9], 0.0, 1.0))
                prior_refinement = state[row, 12:14].astype(np.float32)
                ideal_lead = _ideal_lead(current, future, confidence)
                naive_lead = future[0] - current
                ideal_refinement = _clamp_vector(ideal_lead - naive_lead, 35.0)
                distance = float(np.linalg.norm(current - np.array([state[row, 14] * 0.5, state[row, 14] * 0.5], dtype=np.float32)))
                damping = near_lock_damping(distance, 42.0)
                ideal_refinement *= np.float32(damping)
                targeting_features.append(
                    _cpp_scaled_features(
                        current,
                        float(state[row, 4]),
                        float(state[row, 5]),
                        state[row, 6:8].astype(np.float32),
                        float(state[row, 8]),
                        confidence,
                        prior_refinement,
                        future,
                    )
                )
                targeting_refinement.append(ideal_refinement.astype(np.float32))
                targeting_confidence.append(float(np.clip(confidence * (0.35 + 0.65 * damping), 0.0, 1.0)))

    if not temporal_history:
        raise ValueError("No usable real-world samples were produced from the supplied logs")

    return (
        np.asarray(temporal_history, dtype=np.float32),
        np.asarray(temporal_future, dtype=np.float32),
        np.asarray(targeting_features, dtype=np.float32),
        np.asarray(targeting_refinement, dtype=np.float32),
        np.asarray(targeting_confidence, dtype=np.float32).reshape(-1, 1),
    )


def _write_temporal(path: Path, history: np.ndarray, future: np.ndarray, profiles: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        history=history.astype(np.float32, copy=False),
        future=future.astype(np.float32, copy=False),
        profiles=profiles,
        metadata_json=json.dumps(metadata),
    )


def _write_targeting(path: Path, features: np.ndarray, refinement: np.ndarray, confidence: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float32, copy=False),
        refinement=refinement.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        metadata_json=json.dumps(metadata),
    )


def convert_real_world_logs(
    inputs: list[str | Path],
    temporal_output: str | Path,
    targeting_output: str | Path,
    history_length: int = 12,
    prediction_horizon: int = 16,
    synthetic_temporal: str | Path | None = None,
    synthetic_targeting: str | Path | None = None,
    mix_ratio: float = 1.0,
    heldout_fraction: float = 0.0,
    seed: int = 1337,
) -> dict:
    logs = discover_inputs(inputs)
    if not logs:
        raise ValueError("No real-world .npz logs found")

    real_history, real_future, real_features, real_refinement, real_confidence = build_real_world_datasets(
        logs,
        history_length,
        prediction_horizon,
    )

    history, future, profiles = _mix_temporal(real_history, real_future, synthetic_temporal, mix_ratio, seed)
    features, refinement, confidence = _mix_targeting(
        real_features,
        real_refinement,
        real_confidence,
        synthetic_targeting,
        mix_ratio,
        seed,
    )

    temporal_path = resolve_repo_path(temporal_output)
    targeting_path = resolve_repo_path(targeting_output)
    metadata = {
        "source": "real_world",
        "logs": [str(path) for path in logs],
        "history_length": int(history.shape[1]),
        "prediction_horizon": int(future.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "state_columns": STATE_COLUMNS,
        "mix_ratio_real": float(np.clip(mix_ratio, 0.0, 1.0)),
        "heldout_fraction": float(np.clip(heldout_fraction, 0.0, 0.8)),
    }
    _write_temporal(temporal_path, history, future, profiles, {**metadata, "samples": int(history.shape[0])})
    _write_targeting(
        targeting_path,
        features,
        refinement,
        confidence,
        {
            **metadata,
            "samples": int(features.shape[0]),
            "feature_columns": feature_columns_for_horizon(prediction_horizon),
            "label_columns": LABEL_COLUMNS,
            "input_feature_count": int(features.shape[1]),
        },
    )
    return {
        "temporal_output": str(temporal_path),
        "targeting_output": str(targeting_path),
        "temporal_samples": int(history.shape[0]),
        "targeting_samples": int(features.shape[0]),
        "real_temporal_samples": int(real_history.shape[0]),
        "real_targeting_samples": int(real_features.shape[0]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert real-world runtime logs into training datasets.")
    parser.add_argument("--input", nargs="+", default=["training/datasets/real_world"])
    parser.add_argument("--temporal-output", default="training/datasets/temporal_tracks_realworld.npz")
    parser.add_argument("--targeting-output", default="training/datasets/neural_targeting_tracks_realworld.npz")
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--synthetic-temporal", default=None)
    parser.add_argument("--synthetic-targeting", default=None)
    parser.add_argument("--mix-ratio", type=float, default=1.0, help="Fraction of final samples from real-world data.")
    parser.add_argument("--heldout-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = convert_real_world_logs(
        inputs=args.input,
        temporal_output=args.temporal_output,
        targeting_output=args.targeting_output,
        history_length=args.history_length,
        prediction_horizon=args.prediction_horizon,
        synthetic_temporal=args.synthetic_temporal,
        synthetic_targeting=args.synthetic_targeting,
        mix_ratio=args.mix_ratio,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
