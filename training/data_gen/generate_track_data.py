#!/usr/bin/env python3
"""Generate synthetic temporal track data for the learned predictor."""

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

from training.temporal_predictor.dataset import FEATURE_COLUMNS


PROFILE_NAMES = [
    "constant_velocity",
    "acceleration",
    "curve",
    "abrupt_change",
    "stop_and_go",
    "partial_occlusion",
    "camera_shake",
]


@dataclass(frozen=True)
class TrackDataConfig:
    samples: int = 8192
    history_length: int = 12
    prediction_horizon: int = 16
    detection_width: int = 320
    detection_height: int = 320
    fps: float = 60.0
    seed: int = 1337
    jitter_px: float = 0.35
    camera_shake_px: float = 0.45
    occlusion_probability: float = 0.08
    min_box_size: float = 8.0
    max_box_size: float = 72.0
    max_speed_px_s: float = 1350.0


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def _reflect(value: float, velocity: float, lo: float, hi: float) -> tuple[float, float]:
    if value < lo:
        return lo + (lo - value), abs(velocity)
    if value > hi:
        return hi - (value - hi), -abs(velocity)
    return value, velocity


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-math.pi, math.pi)
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def _simulate_truth(
    rng: np.random.Generator,
    cfg: TrackDataConfig,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_steps = cfg.history_length + cfg.prediction_horizon
    dt = 1.0 / max(1.0, cfg.fps)
    width = float(max(8, cfg.detection_width))
    height = float(max(8, cfg.detection_height))

    position = np.array(
        [
            rng.uniform(width * 0.18, width * 0.82),
            rng.uniform(height * 0.18, height * 0.82),
        ],
        dtype=np.float32,
    )
    speed = rng.uniform(cfg.max_speed_px_s * 0.05, cfg.max_speed_px_s * 0.45)
    velocity = _unit_vector(rng) * speed
    accel = _unit_vector(rng) * rng.uniform(35.0, 420.0)
    turn_rate = rng.uniform(-2.6, 2.6)
    abrupt_step = int(rng.integers(max(2, cfg.history_length // 2), max(3, total_steps - 2)))
    base_size = rng.uniform(cfg.min_box_size, cfg.max_box_size)
    size = np.array([base_size * rng.uniform(0.70, 1.25), base_size * rng.uniform(1.05, 1.80)], dtype=np.float32)
    size_velocity = rng.normal(0.0, 18.0, size=2).astype(np.float32)

    positions = np.zeros((total_steps, 2), dtype=np.float32)
    velocities = np.zeros((total_steps, 2), dtype=np.float32)
    boxes = np.zeros((total_steps, 2), dtype=np.float32)

    for step in range(total_steps):
        phase = step / max(1, total_steps - 1)

        if profile == "acceleration":
            velocity += accel * dt
        elif profile == "curve":
            angle = turn_rate * dt
            c = math.cos(angle)
            s = math.sin(angle)
            velocity = np.array(
                [velocity[0] * c - velocity[1] * s, velocity[0] * s + velocity[1] * c],
                dtype=np.float32,
            )
        elif profile == "abrupt_change" and step == abrupt_step:
            velocity = _unit_vector(rng) * rng.uniform(cfg.max_speed_px_s * 0.20, cfg.max_speed_px_s * 0.65)
        elif profile == "stop_and_go":
            gate = 0.18 + 0.82 * abs(math.sin(phase * math.tau * 1.4 + 0.35))
            velocity += accel * dt * 0.25
            velocity *= np.float32(0.92 + 0.08 * gate)
        elif profile == "partial_occlusion":
            velocity += accel * dt * 0.18
        elif profile == "camera_shake":
            velocity += rng.normal(0.0, 18.0, size=2).astype(np.float32) * dt

        current_speed = float(np.linalg.norm(velocity))
        if current_speed > cfg.max_speed_px_s:
            velocity *= np.float32(cfg.max_speed_px_s / max(1e-6, current_speed))

        position += velocity * dt
        position[0], velocity[0] = _reflect(position[0], velocity[0], 2.0, width - 2.0)
        position[1], velocity[1] = _reflect(position[1], velocity[1], 2.0, height - 2.0)

        size += size_velocity * dt
        for axis in range(2):
            size[axis], size_velocity[axis] = _reflect(
                float(size[axis]),
                float(size_velocity[axis]),
                cfg.min_box_size,
                cfg.max_box_size,
            )

        positions[step] = position
        velocities[step] = velocity
        boxes[step] = size

    return positions, velocities, boxes


def _observed_history(
    rng: np.random.Generator,
    cfg: TrackDataConfig,
    profile: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    history = np.zeros((cfg.history_length, len(FEATURE_COLUMNS)), dtype=np.float32)
    dt = 1.0 / max(1.0, cfg.fps)
    camera_offset = np.zeros(2, dtype=np.float32)
    held_position = positions[0].copy()

    for step in range(cfg.history_length):
        confidence = float(np.clip(rng.normal(0.88, 0.10), 0.15, 1.0))
        jitter_scale = cfg.jitter_px

        if profile == "camera_shake":
            camera_offset = camera_offset * 0.72 + rng.normal(0.0, cfg.camera_shake_px, size=2)
            jitter_scale += cfg.camera_shake_px * 0.50
        else:
            camera_offset *= 0.65

        occluded = profile == "partial_occlusion" and rng.random() < cfg.occlusion_probability
        if occluded:
            confidence = float(rng.uniform(0.05, 0.35))
            held_position = held_position * 0.82 + positions[step] * 0.18
            observed_pos = held_position + rng.normal(0.0, jitter_scale * 2.2, size=2)
        else:
            observed_pos = positions[step] + camera_offset + rng.normal(0.0, jitter_scale, size=2)
            held_position = observed_pos.astype(np.float32)

        observed_box = boxes[step] * rng.normal(1.0, 0.025, size=2)
        observed_box = np.clip(observed_box, cfg.min_box_size, cfg.max_box_size)
        if step == 0:
            box_scale_vel = 0.0
        else:
            box_scale_vel = (
                float(np.mean(observed_box)) - float(np.mean(history[step - 1, 2:4]))
            ) / dt

        history[step] = np.array(
            [
                observed_pos[0],
                observed_pos[1],
                observed_box[0],
                observed_box[1],
                velocities[step, 0],
                velocities[step, 1],
                box_scale_vel,
                confidence,
            ],
            dtype=np.float32,
        )

    return history


def generate_dataset(cfg: TrackDataConfig):
    cfg = TrackDataConfig(
        samples=max(1, int(cfg.samples)),
        history_length=max(2, int(cfg.history_length)),
        prediction_horizon=max(1, int(cfg.prediction_horizon)),
        detection_width=max(8, int(cfg.detection_width)),
        detection_height=max(8, int(cfg.detection_height)),
        fps=max(1.0, float(cfg.fps)),
        seed=int(cfg.seed),
        jitter_px=max(0.0, float(cfg.jitter_px)),
        camera_shake_px=max(0.0, float(cfg.camera_shake_px)),
        occlusion_probability=float(np.clip(cfg.occlusion_probability, 0.0, 1.0)),
        min_box_size=max(1.0, float(cfg.min_box_size)),
        max_box_size=max(float(cfg.min_box_size) + 1.0, float(cfg.max_box_size)),
        max_speed_px_s=max(1.0, float(cfg.max_speed_px_s)),
    )

    rng = np.random.default_rng(cfg.seed)
    history = np.zeros((cfg.samples, cfg.history_length, len(FEATURE_COLUMNS)), dtype=np.float32)
    future = np.zeros((cfg.samples, cfg.prediction_horizon, 2), dtype=np.float32)
    profiles = np.empty((cfg.samples,), dtype="<U24")

    for sample_idx in range(cfg.samples):
        profile = PROFILE_NAMES[sample_idx % len(PROFILE_NAMES)]
        if sample_idx >= len(PROFILE_NAMES):
            profile = str(rng.choice(PROFILE_NAMES))

        positions, velocities, boxes = _simulate_truth(rng, cfg, profile)
        history[sample_idx] = _observed_history(rng, cfg, profile, positions, velocities, boxes)
        start = cfg.history_length
        future[sample_idx] = positions[start : start + cfg.prediction_horizon]
        profiles[sample_idx] = profile

    metadata = {
        **asdict(cfg),
        "feature_columns": FEATURE_COLUMNS,
        "profile_names": PROFILE_NAMES,
    }
    return history, future, profiles, metadata


def save_dataset(output: str | Path, cfg: TrackDataConfig) -> Path:
    output_path = resolve_repo_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history, future, profiles, metadata = generate_dataset(cfg)
    np.savez_compressed(
        output_path,
        history=history,
        future=future,
        profiles=profiles,
        metadata_json=json.dumps(metadata),
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic temporal track data.")
    parser.add_argument("--output", default="training/datasets/temporal_tracks.npz")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--detection-width", type=int, default=320)
    parser.add_argument("--detection-height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jitter-px", type=float, default=0.35)
    parser.add_argument("--camera-shake-px", type=float, default=0.45)
    parser.add_argument("--occlusion-probability", type=float, default=0.08)
    parser.add_argument("--max-speed-px-s", type=float, default=1350.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = save_dataset(
        args.output,
        TrackDataConfig(
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
        ),
    )
    print(f"Saved temporal track dataset to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
