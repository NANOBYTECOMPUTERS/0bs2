#!/usr/bin/env python3
"""Generate synthetic data for the advisory neural targeting head."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data_gen.generate_track_data import TrackDataConfig, generate_dataset
from training.neural_targeting.dataset import BASE_FEATURE_COLUMNS, LABEL_COLUMNS, feature_columns_for_horizon


@dataclass(frozen=True)
class TargetingDataConfig:
    samples: int = 8192
    history_length: int = 12
    prediction_horizon: int = 16
    detection_width: int = 320
    detection_height: int = 320
    fps: float = 60.0
    seed: int = 4242
    jitter_px: float = 0.35
    camera_shake_px: float = 0.45
    occlusion_probability: float = 0.08
    max_speed_px_s: float = 1350.0
    max_refinement_px: float = 35.0
    prediction_noise_px: float = 2.75
    prediction_bias_px: float = 4.50
    bad_prediction_probability: float = 0.14
    near_lock_radius_px: float = 60.0


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def near_lock_damping(distance_px: float, radius_px: float) -> float:
    return _smoothstep(distance_px / max(1.0, radius_px))


def _clamp_vector(vec: np.ndarray, max_len: float) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= max_len or length <= 1e-6:
        return np.nan_to_num(vec, nan=0.0).astype(np.float32)
    return (vec * (max_len / length)).astype(np.float32)


def _cpp_scaled_features(
    center: np.ndarray,
    width: float,
    height: float,
    velocity: np.ndarray,
    box_scale_vel: float,
    confidence: float,
    prior_refinement: np.ndarray,
    predicted_future: np.ndarray,
) -> np.ndarray:
    values: list[float] = [
        float(np.clip(center[0] / 640.0, -4.0, 4.0)),
        float(np.clip(center[1] / 640.0, -4.0, 4.0)),
        float(np.clip(width / 320.0, -4.0, 4.0)),
        float(np.clip(height / 320.0, -4.0, 4.0)),
        float(np.clip(velocity[0] / 1500.0, -4.0, 4.0)),
        float(np.clip(velocity[1] / 1500.0, -4.0, 4.0)),
        float(np.clip(box_scale_vel / 800.0, -4.0, 4.0)),
        float(np.clip(confidence, 0.0, 1.0)),
        float(np.clip(prior_refinement[0] / 80.0, -4.0, 4.0)),
        float(np.clip(prior_refinement[1] / 80.0, -4.0, 4.0)),
    ]
    for point in predicted_future:
        values.append(float(np.clip((point[0] - center[0]) / 640.0, -4.0, 4.0)))
        values.append(float(np.clip((point[1] - center[1]) / 640.0, -4.0, 4.0)))
    return np.asarray(values, dtype=np.float32)


def _ideal_lead(current: np.ndarray, future: np.ndarray, confidence: float) -> np.ndarray:
    horizon = future.shape[0]
    near_count = max(1, min(horizon, 4))
    weights = np.linspace(1.0, 0.35, near_count, dtype=np.float32)
    weights /= np.sum(weights)
    intercept = np.sum(future[:near_count] * weights.reshape(-1, 1), axis=0)
    lead = intercept - current
    if confidence < 0.45:
        lead *= np.float32(0.35 + confidence)
    return lead.astype(np.float32)


def generate_targeting_dataset(cfg: TargetingDataConfig):
    cfg = TargetingDataConfig(
        samples=max(2, int(cfg.samples)),
        history_length=max(2, int(cfg.history_length)),
        prediction_horizon=max(1, int(cfg.prediction_horizon)),
        detection_width=max(8, int(cfg.detection_width)),
        detection_height=max(8, int(cfg.detection_height)),
        fps=max(1.0, float(cfg.fps)),
        seed=int(cfg.seed),
        jitter_px=max(0.0, float(cfg.jitter_px)),
        camera_shake_px=max(0.0, float(cfg.camera_shake_px)),
        occlusion_probability=float(np.clip(cfg.occlusion_probability, 0.0, 1.0)),
        max_speed_px_s=max(1.0, float(cfg.max_speed_px_s)),
        max_refinement_px=max(1.0, float(cfg.max_refinement_px)),
        prediction_noise_px=max(0.0, float(cfg.prediction_noise_px)),
        prediction_bias_px=max(0.0, float(cfg.prediction_bias_px)),
        bad_prediction_probability=float(np.clip(cfg.bad_prediction_probability, 0.0, 1.0)),
        near_lock_radius_px=max(1.0, float(cfg.near_lock_radius_px)),
    )
    rng = np.random.default_rng(cfg.seed)

    history, future, profiles, temporal_metadata = generate_dataset(
        TrackDataConfig(
            samples=cfg.samples,
            history_length=cfg.history_length,
            prediction_horizon=cfg.prediction_horizon,
            detection_width=cfg.detection_width,
            detection_height=cfg.detection_height,
            fps=cfg.fps,
            seed=cfg.seed,
            jitter_px=cfg.jitter_px,
            camera_shake_px=cfg.camera_shake_px,
            occlusion_probability=cfg.occlusion_probability,
            max_speed_px_s=cfg.max_speed_px_s,
        )
    )

    feature_dim = len(BASE_FEATURE_COLUMNS) + cfg.prediction_horizon * 2
    features = np.zeros((cfg.samples, feature_dim), dtype=np.float32)
    refinement = np.zeros((cfg.samples, 2), dtype=np.float32)
    confidence = np.zeros((cfg.samples, 1), dtype=np.float32)

    screen_center = np.array([cfg.detection_width * 0.5, cfg.detection_height * 0.5], dtype=np.float32)

    for i in range(cfg.samples):
        current = history[i, -1, 0:2].astype(np.float32)
        width = float(history[i, -1, 2])
        height = float(history[i, -1, 3])
        velocity = history[i, -1, 4:6].astype(np.float32)
        box_scale_vel = float(history[i, -1, 6])
        track_confidence = float(np.clip(history[i, -1, 7], 0.0, 1.0))

        temporal_bias = rng.normal(0.0, cfg.prediction_bias_px, size=2).astype(np.float32)
        temporal_bias *= np.float32(rng.uniform(0.0, 1.0))
        predicted_future = future[i].astype(np.float32).copy()
        predicted_future += temporal_bias
        predicted_future += rng.normal(0.0, cfg.prediction_noise_px, size=predicted_future.shape).astype(np.float32)

        bad_prediction = rng.random() < cfg.bad_prediction_probability
        if bad_prediction:
            wrong_direction = rng.uniform(-math.pi, math.pi)
            wrong_offset = np.array([math.cos(wrong_direction), math.sin(wrong_direction)], dtype=np.float32)
            wrong_offset *= np.float32(rng.uniform(18.0, 70.0))
            predicted_future += wrong_offset

        ideal_lead = _ideal_lead(current, future[i], track_confidence)
        naive_lead = predicted_future[0] - current
        ideal_refinement = ideal_lead - naive_lead

        distance_to_crosshair = float(np.linalg.norm(current - screen_center))
        damping = near_lock_damping(distance_to_crosshair, cfg.near_lock_radius_px)
        ideal_refinement *= np.float32(damping)
        if bad_prediction:
            ideal_refinement *= np.float32(0.20)

        ideal_refinement = _clamp_vector(ideal_refinement, cfg.max_refinement_px)
        prior_refinement = ideal_refinement * np.float32(rng.uniform(-0.25, 0.65))
        if rng.random() < 0.55:
            prior_refinement[:] = 0.0

        prediction_error = float(np.linalg.norm((predicted_future[0] - future[i, 0]).astype(np.float32)))
        confidence_label = track_confidence
        confidence_label *= 1.0 - 0.75 * float(bad_prediction)
        confidence_label *= 1.0 - np.clip(prediction_error / 80.0, 0.0, 0.75)
        confidence_label *= 0.35 + 0.65 * damping
        confidence_label = float(np.clip(confidence_label, 0.0, 1.0))

        features[i] = _cpp_scaled_features(
            current,
            width,
            height,
            velocity,
            box_scale_vel,
            track_confidence,
            prior_refinement,
            predicted_future,
        )
        refinement[i] = ideal_refinement
        confidence[i, 0] = confidence_label

    profile_names, profile_counts = np.unique(profiles, return_counts=True)
    metadata = {
        **asdict(cfg),
        "feature_columns": feature_columns_for_horizon(cfg.prediction_horizon),
        "label_columns": LABEL_COLUMNS,
        "input_feature_count": feature_dim,
        "cpp_feature_scaling": {
            "center": 640.0,
            "box_size": 320.0,
            "velocity": 1500.0,
            "box_scale_vel": 800.0,
            "refinement": 80.0,
            "future_delta": 640.0,
        },
        "temporal_profiles": temporal_metadata.get("profile_names", []),
        "profile_counts": {
            str(name): int(count)
            for name, count in zip(profile_names.tolist(), profile_counts.tolist())
        },
    }
    return features, refinement, confidence, metadata


def save_dataset(output: str | Path, cfg: TargetingDataConfig) -> Path:
    output_path = resolve_repo_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features, refinement, confidence, metadata = generate_targeting_dataset(cfg)
    np.savez_compressed(
        output_path,
        features=features,
        refinement=refinement,
        confidence=confidence,
        metadata_json=json.dumps(metadata),
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic neural targeting head data.")
    parser.add_argument("--output", default="training/datasets/neural_targeting_tracks.npz")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--detection-width", type=int, default=320)
    parser.add_argument("--detection-height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--jitter-px", type=float, default=0.35)
    parser.add_argument("--camera-shake-px", type=float, default=0.45)
    parser.add_argument("--occlusion-probability", type=float, default=0.08)
    parser.add_argument("--max-speed-px-s", type=float, default=1350.0)
    parser.add_argument("--max-refinement-px", type=float, default=35.0)
    parser.add_argument("--prediction-noise-px", type=float, default=2.75)
    parser.add_argument("--prediction-bias-px", type=float, default=4.50)
    parser.add_argument("--bad-prediction-probability", type=float, default=0.14)
    parser.add_argument("--near-lock-radius-px", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = save_dataset(
        args.output,
        TargetingDataConfig(
            samples=args.samples,
            history_length=args.history_length,
            prediction_horizon=args.prediction_horizon,
            detection_width=args.detection_width,
            detection_height=args.detection_height,
            fps=args.fps,
            seed=args.seed,
            jitter_px=args.jitter_px,
            camera_shake_px=args.camera_shake_px,
            occlusion_probability=args.occlusion_probability,
            max_speed_px_s=args.max_speed_px_s,
            max_refinement_px=args.max_refinement_px,
            prediction_noise_px=args.prediction_noise_px,
            prediction_bias_px=args.prediction_bias_px,
            bad_prediction_probability=args.bad_prediction_probability,
            near_lock_radius_px=args.near_lock_radius_px,
        ),
    )
    print(f"Saved neural targeting dataset to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
