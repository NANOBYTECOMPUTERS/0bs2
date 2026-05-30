#!/usr/bin/env python3
"""
Synthetic dataset generator for a learned PID speed/gain governor.

The generated labels do not command raw mouse deltas. They teach a small model
to scale PID terms and final speed so the C++ controller can keep deterministic
fallback behavior and hard safety limits.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


DIRECTION_COLUMNS = [
    "error_direction_right",
    "error_direction_down_right",
    "error_direction_down",
    "error_direction_down_left",
    "error_direction_left",
    "error_direction_up_left",
    "error_direction_up",
    "error_direction_up_right",
]

MOTION_STATE_COLUMNS = [
    "target_motion_still",
    "target_motion_moving",
]

# Exact order must match C++ pidGovernorFeatures() emission order (40 features).
# See mouse/PidGovernor.cpp and mouse/PidGovernor.h (PidGovernorFeatureCount = 40).
FEATURE_COLUMNS = [
    "error_x",
    "error_y",
    "error_distance",
    *DIRECTION_COLUMNS,
    "target_width",
    "target_height",
    "target_size",
    "target_offset_x",
    "target_offset_y",
    "aim_point_error_x",
    "aim_point_error_y",
    "target_vx",
    "target_vy",
    "target_ax",
    "target_ay",
    *MOTION_STATE_COLUMNS,
    "cursor_vx",
    "cursor_vy",
    "previous_output_x",
    "previous_output_y",
    "pid_p_x",
    "pid_p_y",
    "pid_i_x",
    "pid_i_y",
    "pid_d_x",
    "pid_d_y",
    "closing_rate",
    "overshoot_risk",
    "dt",
    "confidence",
    "max_speed_ratio",
    "box_aspect_ratio",
]

LABEL_COLUMNS = [
    "label_kp_scale",
    "label_ki_scale",
    "label_kd_scale",
    "label_speed_scale",
]

PROFILE_NAMES = [
    "stopped_center",
    "stopped_offset",
    "linear_tracking",
    "direction_change",
    "moving_away",
    "jitter_and_loss",
    "stop_and_go",
]


@dataclass(frozen=True)
class PidConfig:
    actuator_hz: int = 2000
    kp: float = 0.0250
    ki: float = 0.0003
    kd: float = 0.0001
    max_pixel_step: float = 10.0
    output_scale: float = 0.10
    max_integral: float = 120.0
    derivative_tau_ms: float = 18.0
    size_reference_px: float = 640.0

    @property
    def nominal_speed_px_s(self) -> float:
        return max(0.01, self.max_pixel_step) * max(1, self.actuator_hz) * max(0.01, self.output_scale)


@dataclass(frozen=True)
class DatasetConfig:
    episodes_per_profile: int = 64
    steps_per_episode: int = 180
    seed: int = 1337
    max_speed_multiple: float = 5.0
    min_target_size_px: float = 6.0
    max_target_size_px: float = 96.0
    detection_noise_px: float = 0.35
    dt_jitter_fraction: float = 0.10
    max_target_speed_px_s: float | None = None

    def speed_cap_px_s(self, pid: PidConfig) -> float:
        learned_cap = pid.nominal_speed_px_s * self.max_speed_multiple
        if self.max_target_speed_px_s is None or self.max_target_speed_px_s <= 0.0:
            return learned_cap
        return min(learned_cap, self.max_target_speed_px_s)


def _float_or_default(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_or_default(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_scalar_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_pid_config(path: Path) -> PidConfig:
    values = read_scalar_config(path)
    return PidConfig(
        actuator_hz=_int_or_default(values.get("pid_actuator_hz"), 2000),
        kp=_float_or_default(values.get("pid_kp"), 0.0250),
        ki=_float_or_default(values.get("pid_ki"), 0.0003),
        kd=_float_or_default(values.get("pid_kd"), 0.0001),
        max_pixel_step=_float_or_default(values.get("pid_max_pixel_step"), 10.0),
        output_scale=_float_or_default(values.get("pid_output_scale"), 0.10),
        max_integral=_float_or_default(values.get("pid_max_integral"), 120.0),
        derivative_tau_ms=_float_or_default(values.get("pid_derivative_filter_tau_ms"), 18.0),
        size_reference_px=_float_or_default(values.get("pid_size_reference_px"), 48.0),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _smoothstep(value: float) -> float:
    t = _clamp(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _safe_hypot(x: float, y: float) -> float:
    return math.hypot(float(x), float(y))


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % math.tau - math.pi)


def _eight_axis_direction_features(error_x: float, error_y: float) -> dict[str, float]:
    if not math.isfinite(error_x) or not math.isfinite(error_y):
        return {column: 1.0 / len(DIRECTION_COLUMNS) for column in DIRECTION_COLUMNS}

    distance = _safe_hypot(error_x, error_y)
    if distance < 1e-6:
        return {column: 1.0 / len(DIRECTION_COLUMNS) for column in DIRECTION_COLUMNS}

    angle = math.atan2(error_y, error_x)
    centers = [
        0.0,
        math.pi / 4.0,
        math.pi / 2.0,
        3.0 * math.pi / 4.0,
        math.pi,
        -3.0 * math.pi / 4.0,
        -math.pi / 2.0,
        -math.pi / 4.0,
    ]
    sector_width = math.pi / 4.0
    weights = [max(0.0, 1.0 - _angle_delta(angle, center) / sector_width) for center in centers]
    total = sum(weights)
    if total <= 1e-9 or not math.isfinite(total):
        weights = [1.0 / len(DIRECTION_COLUMNS)] * len(DIRECTION_COLUMNS)
    else:
        weights = [weight / total for weight in weights]

    return dict(zip(DIRECTION_COLUMNS, weights))


def _motion_state_features(target_vx: float, target_vy: float, target_size: float) -> dict[str, float]:
    if not math.isfinite(target_vx) or not math.isfinite(target_vy):
        return {"target_motion_still": 1.0, "target_motion_moving": 0.0}

    speed = _safe_hypot(target_vx, target_vy)
    size = max(1.0, target_size if math.isfinite(target_size) else 1.0)
    still_threshold = max(0.75, size * 0.08)
    moving_threshold = max(still_threshold + 0.75, size * 0.65)
    moving = _smoothstep((speed - still_threshold) / max(1e-6, moving_threshold - still_threshold))
    return {
        "target_motion_still": 1.0 - moving,
        "target_motion_moving": moving,
    }


def compute_teacher_labels(features: dict[str, float], pid: PidConfig, cfg: DatasetConfig) -> dict[str, float]:
    """Return supervised labels for PID term scales and final speed scale.

    The teacher is intentionally conservative near center, especially for small
    targets. It increases authority when distance is growing, and it brakes when
    overshoot risk appears.
    """

    # Support both legacy long names and the current 40-col short names for robustness.
    def _get(name: str, default: float = 0.0) -> float:
        if name in features:
            return float(features.get(name, default))
        # legacy fallbacks
        legacy = {
            "error_distance": "error_distance_px",
            "target_size": "target_size_px",
            "closing_rate": "closing_rate_px_s",
            "target_vx": "target_speed_x_px_s",
            "target_vy": "target_speed_y_px_s",
        }.get(name)
        if legacy and legacy in features:
            return float(features.get(legacy, default))
        return float(features.get(name, default))

    distance = max(0.0, _get("error_distance", 0.0))
    target_size = max(1.0, _get("target_size", pid.size_reference_px))
    closing_rate = _get("closing_rate", 0.0)
    overshoot_risk = _clamp(_get("overshoot_risk", 0.0), 0.0, 1.0)
    confidence = _clamp(_get("confidence", 1.0), 0.0, 1.0)
    diagonal_weight = _clamp(
        _get("error_direction_down_right", 0.0)
        + _get("error_direction_down_left", 0.0)
        + _get("error_direction_up_left", 0.0)
        + _get("error_direction_up_right", 0.0),
        0.0,
        1.0,
    )
    still_weight = _clamp(_get("target_motion_still", 0.0), 0.0, 1.0)
    moving_weight = _clamp(_get("target_motion_moving", 0.0), 0.0, 1.0)

    precision_radius = max(0.10, target_size * 0.018)
    slowdown_radius = max(precision_radius + 0.50, target_size * 0.65)
    distance_t = (distance - precision_radius) / max(0.001, slowdown_radius - precision_radius)
    near_factor = _smoothstep(distance_t)

    small_target_scale = _clamp(target_size / max(1.0, pid.size_reference_px), 0.16, 1.0)
    speed_scale = near_factor * small_target_scale

    if closing_rate < -2.0:
        diverging = _clamp(abs(closing_rate) / max(1.0, pid.nominal_speed_px_s), 0.0, 1.0)
        speed_scale *= 1.0 + 0.70 * diverging
    elif closing_rate > 2.0:
        approaching_fast = _clamp(closing_rate / max(1.0, pid.nominal_speed_px_s), 0.0, 1.0)
        speed_scale *= 1.0 - 0.55 * approaching_fast

    speed_scale *= 1.0 - 0.78 * overshoot_risk
    speed_scale *= 1.0 - 0.08 * diagonal_weight
    speed_scale *= 0.82 + 0.18 * moving_weight
    speed_scale *= 0.20 + 0.80 * confidence
    speed_scale = _clamp(speed_scale, 0.0, 1.0)

    kp_scale = _clamp(0.10 + 0.95 * near_factor, 0.05, 1.0)
    kp_scale *= 0.70 + 0.30 * small_target_scale
    if closing_rate < -2.0:
        kp_scale *= 1.0 + 0.25 * _clamp(abs(closing_rate) / max(1.0, pid.nominal_speed_px_s), 0.0, 1.0)
    kp_scale *= 1.0 - 0.55 * overshoot_risk
    kp_scale *= 1.0 - 0.03 * diagonal_weight
    kp_scale *= 0.88 + 0.12 * moving_weight
    kp_scale *= 0.35 + 0.65 * confidence

    ki_scale = _clamp((near_factor - 0.20) / 0.80, 0.0, 1.0)
    ki_scale *= 1.0 - 0.90 * overshoot_risk
    if distance <= slowdown_radius:
        ki_scale *= 0.25
    ki_scale *= 1.0 - 0.45 * moving_weight
    ki_scale *= confidence

    kd_scale = 0.18 + 0.65 * overshoot_risk
    if closing_rate > 2.0:
        kd_scale += 0.35 * _clamp(closing_rate / max(1.0, pid.nominal_speed_px_s), 0.0, 1.0)
    kd_scale += 0.15 * (1.0 - small_target_scale)
    kd_scale += 0.08 * diagonal_weight
    kd_scale += 0.06 * still_weight
    kd_scale *= 0.50 + 0.50 * confidence

    return {
        "label_kp_scale": _clamp(kp_scale, 0.0, 1.0),
        "label_ki_scale": _clamp(ki_scale, 0.0, 1.0),
        "label_kd_scale": _clamp(kd_scale, 0.0, 1.0),
        "label_speed_scale": speed_scale,
    }


def _random_unit_vector(rng: random.Random) -> tuple[float, float]:
    angle = rng.uniform(0.0, math.tau)
    return math.cos(angle), math.sin(angle)


def _random_error(rng: random.Random, min_distance: float, max_distance: float) -> tuple[float, float]:
    ux, uy = _random_unit_vector(rng)
    distance = rng.uniform(min_distance, max_distance)
    return ux * distance, uy * distance


def _random_size(rng: random.Random, cfg: DatasetConfig) -> tuple[float, float]:
    size = math.exp(rng.uniform(math.log(cfg.min_target_size_px), math.log(cfg.max_target_size_px)))
    aspect = rng.uniform(0.55, 1.45)
    width = max(2.0, size * math.sqrt(aspect))
    height = max(2.0, size / math.sqrt(aspect))
    return width, height


def _profile_initial_state(profile: str, rng: random.Random, pid: PidConfig, cfg: DatasetConfig) -> dict[str, float]:
    max_speed = cfg.speed_cap_px_s(pid)
    width, height = _random_size(rng, cfg)
    target_size = math.sqrt(width * height)

    if profile == "stopped_center":
        error_x, error_y = _random_error(rng, 0.0, max(0.30, target_size * 0.085))
        vx = vy = ax = ay = 0.0
    elif profile == "stopped_offset":
        error_x, error_y = _random_error(rng, target_size * 0.20, target_size * 2.25 + 24.0)
        vx = vy = ax = ay = 0.0
    elif profile == "linear_tracking":
        error_x, error_y = _random_error(rng, target_size * 0.35, target_size * 2.70 + 32.0)
        ux, uy = _random_unit_vector(rng)
        speed = rng.uniform(max_speed * 0.04, max_speed)
        vx, vy = ux * speed, uy * speed
        ax = ay = 0.0
    elif profile == "direction_change":
        error_x, error_y = _random_error(rng, target_size * 0.40, target_size * 3.00 + 48.0)
        ux, uy = _random_unit_vector(rng)
        speed = rng.uniform(max_speed * 0.08, max_speed * 0.90)
        vx, vy = ux * speed, uy * speed
        ax = ay = 0.0
    elif profile == "moving_away":
        error_x, error_y = _random_error(rng, target_size * 0.30, target_size * 2.60 + 36.0)
        ux, uy = _random_unit_vector(rng)
        speed = rng.uniform(max_speed * 0.04, max_speed * 0.70)
        vx, vy = ux * speed, uy * speed
        ax = ay = 0.0
    elif profile == "jitter_and_loss":
        error_x, error_y = _random_error(rng, 0.0, target_size * 2.00 + 28.0)
        ux, uy = _random_unit_vector(rng)
        speed = rng.uniform(0.0, max_speed * 0.65)
        vx, vy = ux * speed, uy * speed
        ax = ay = 0.0
    elif profile == "stop_and_go":
        error_x, error_y = _random_error(rng, target_size * 0.10, target_size * 2.40 + 32.0)
        vx = vy = ax = ay = 0.0
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return {
        "error_x": error_x,
        "error_y": error_y,
        "target_width": width,
        "target_height": height,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
    }


def _pid_step(
    error_x: float,
    error_y: float,
    prev_error_x: float,
    prev_error_y: float,
    integral_x: float,
    integral_y: float,
    filt_deriv_x: float,
    filt_deriv_y: float,
    dt: float,
    pid: PidConfig,
) -> tuple[dict[str, float], float, float, float, float]:
    derivative_x = (error_x - prev_error_x) / max(1e-6, dt)
    derivative_y = (error_y - prev_error_y) / max(1e-6, dt)
    tau = max(1e-6, pid.derivative_tau_ms * 0.001)
    alpha = _clamp(dt / (dt + tau), 0.0, 1.0)
    filt_deriv_x += (derivative_x - filt_deriv_x) * alpha
    filt_deriv_y += (derivative_y - filt_deriv_y) * alpha

    integral_x = _clamp(integral_x + error_x * dt, -pid.max_integral, pid.max_integral)
    integral_y = _clamp(integral_y + error_y * dt, -pid.max_integral, pid.max_integral)

    terms = {
        "pid_p_x": pid.kp * error_x,
        "pid_p_y": pid.kp * error_y,
        "pid_i_x": pid.ki * integral_x,
        "pid_i_y": pid.ki * integral_y,
        "pid_d_x": pid.kd * filt_deriv_x,
        "pid_d_y": pid.kd * filt_deriv_y,
    }
    return terms, integral_x, integral_y, filt_deriv_x, filt_deriv_y


def _simulate_profile(profile: str, episode: int, rng: random.Random, pid: PidConfig, cfg: DatasetConfig) -> list[dict[str, float | str | int]]:
    state = _profile_initial_state(profile, rng, pid, cfg)
    rows: list[dict[str, float | str | int]] = []
    max_target_speed = cfg.speed_cap_px_s(pid)
    base_dt = 1.0 / max(1, pid.actuator_hz)

    prev_observed_x = state["error_x"]
    prev_observed_y = state["error_y"]
    prev_distance = _safe_hypot(prev_observed_x, prev_observed_y)
    integral_x = integral_y = 0.0
    filt_deriv_x = filt_deriv_y = 0.0
    prev_output_x = prev_output_y = 0.0
    cursor_speed_x = cursor_speed_y = 0.0

    for step in range(cfg.steps_per_episode):
        dt = base_dt * rng.uniform(1.0 - cfg.dt_jitter_fraction, 1.0 + cfg.dt_jitter_fraction)
        progress = step / max(1, cfg.steps_per_episode - 1)

        if profile == "direction_change" and step == cfg.steps_per_episode // 2:
            state["vx"] *= rng.uniform(-1.25, -0.45)
            state["vy"] *= rng.uniform(-1.25, -0.45)

        if profile == "stop_and_go":
            if progress < 0.22:
                state["vx"] = 0.0
                state["vy"] = 0.0
                state["ax"] = 0.0
                state["ay"] = 0.0
            elif progress < 0.62:
                if _safe_hypot(state["vx"], state["vy"]) < max_target_speed * 0.03:
                    ux, uy = _random_unit_vector(rng)
                    speed = rng.uniform(max_target_speed * 0.12, max_target_speed * 0.70)
                    state["vx"] = ux * speed
                    state["vy"] = uy * speed
                state["ax"] = rng.uniform(-max_target_speed * 0.35, max_target_speed * 0.35)
                state["ay"] = rng.uniform(-max_target_speed * 0.35, max_target_speed * 0.35)
            elif progress < 0.80:
                state["vx"] *= 0.90
                state["vy"] *= 0.90
                if _safe_hypot(state["vx"], state["vy"]) < max_target_speed * 0.02:
                    state["vx"] = 0.0
                    state["vy"] = 0.0
                state["ax"] = 0.0
                state["ay"] = 0.0
            else:
                if step % 11 == 0:
                    ux, uy = _random_unit_vector(rng)
                    speed = rng.uniform(max_target_speed * 0.08, max_target_speed * 0.45)
                    state["vx"] = ux * speed
                    state["vy"] = uy * speed
                state["ax"] = rng.uniform(-max_target_speed * 0.20, max_target_speed * 0.20)
                state["ay"] = rng.uniform(-max_target_speed * 0.20, max_target_speed * 0.20)

        if profile == "moving_away":
            shrink = 1.0 - 0.45 * progress
            state["target_width"] = max(cfg.min_target_size_px * 0.75, state["target_width"] * (1.0 - 0.0025))
            state["target_height"] = max(cfg.min_target_size_px * 0.75, state["target_height"] * (1.0 - 0.0025))
            state["vx"] *= 1.0 + 0.0015 * shrink
            state["vy"] *= 1.0 + 0.0015 * shrink

        if profile == "jitter_and_loss":
            state["ax"] = rng.uniform(-max_target_speed * 2.0, max_target_speed * 2.0)
            state["ay"] = rng.uniform(-max_target_speed * 2.0, max_target_speed * 2.0)
        elif profile in {"linear_tracking", "direction_change"}:
            state["ax"] = rng.uniform(-max_target_speed * 0.25, max_target_speed * 0.25)
            state["ay"] = rng.uniform(-max_target_speed * 0.25, max_target_speed * 0.25)

        state["vx"] = _clamp(state["vx"] + state["ax"] * dt, -max_target_speed, max_target_speed)
        state["vy"] = _clamp(state["vy"] + state["ay"] * dt, -max_target_speed, max_target_speed)
        state["error_x"] += state["vx"] * dt
        state["error_y"] += state["vy"] * dt

        confidence = 1.0
        noise = cfg.detection_noise_px
        if profile == "jitter_and_loss":
            confidence = 0.0 if rng.random() < 0.08 else rng.uniform(0.45, 1.0)
            noise *= rng.uniform(1.0, 5.0)
        elif rng.random() < 0.015:
            confidence = rng.uniform(0.55, 0.90)

        observed_x = state["error_x"] + rng.gauss(0.0, noise)
        observed_y = state["error_y"] + rng.gauss(0.0, noise)
        distance = _safe_hypot(observed_x, observed_y)
        target_size = math.sqrt(max(1.0, state["target_width"] * state["target_height"]))
        closing_rate = (prev_distance - distance) / max(1e-6, dt)
        overshoot_risk = 1.0 if (prev_observed_x * observed_x + prev_observed_y * observed_y) < 0.0 else 0.0
        if distance < max(0.15, target_size * 0.08) and closing_rate > pid.nominal_speed_px_s * 0.25:
            overshoot_risk = max(overshoot_risk, 0.70)

        terms, integral_x, integral_y, filt_deriv_x, filt_deriv_y = _pid_step(
            observed_x,
            observed_y,
            prev_observed_x,
            prev_observed_y,
            integral_x,
            integral_y,
            filt_deriv_x,
            filt_deriv_y,
            dt,
            pid,
        )

        features = {
            # Core features (both short and legacy names for maximum compatibility)
            "error_x": float(observed_x),
            "error_y": float(observed_y),
            "error_distance": float(distance),
            **_eight_axis_direction_features(observed_x, observed_y),
            "target_width": float(state["target_width"]),
            "target_height": float(state["target_height"]),
            "target_size": float(target_size),
            "target_vx": float(state["vx"]),
            "target_vy": float(state["vy"]),
            "target_ax": float(state["ax"]),
            "target_ay": float(state["ay"]),
            **_motion_state_features(state["vx"], state["vy"], target_size),
            "cursor_vx": float(cursor_speed_x),
            "cursor_vy": float(cursor_speed_y),
            "previous_output_x": float(prev_output_x),
            "previous_output_y": float(prev_output_y),
            **terms,
            "closing_rate": float(closing_rate),
            "overshoot_risk": float(overshoot_risk),
            "dt": float(dt),
            "confidence": float(confidence),
            "max_speed_ratio": float(cfg.max_speed_multiple),
            # Legacy long names (teacher + old tooling)
            "error_x_px": float(observed_x),
            "error_y_px": float(observed_y),
            "error_distance_px": float(distance),
            "target_width_px": float(state["target_width"]),
            "target_height_px": float(state["target_height"]),
            "target_size_px": float(target_size),
            "target_speed_x_px_s": float(state["vx"]),
            "target_speed_y_px_s": float(state["vy"]),
            "target_accel_x_px_s2": float(state["ax"]),
            "target_accel_y_px_s2": float(state["ay"]),
            "cursor_speed_x_px_s": float(cursor_speed_x),
            "cursor_speed_y_px_s": float(cursor_speed_y),
            "previous_output_x_px": float(prev_output_x),
            "previous_output_y_px": float(prev_output_y),
            "closing_rate_px_s": float(closing_rate),
            "dt_s": float(dt),
            "max_speed_ratio": float(cfg.max_speed_multiple),
        }
        labels = compute_teacher_labels(features, pid, cfg)

        raw_x = (
            terms["pid_p_x"] * labels["label_kp_scale"]
            + terms["pid_i_x"] * labels["label_ki_scale"]
            + terms["pid_d_x"] * labels["label_kd_scale"]
        )
        raw_y = (
            terms["pid_p_y"] * labels["label_kp_scale"]
            + terms["pid_i_y"] * labels["label_ki_scale"]
            + terms["pid_d_y"] * labels["label_kd_scale"]
        )
        out_mag = _safe_hypot(raw_x, raw_y)
        max_step = pid.max_pixel_step * cfg.max_speed_multiple * labels["label_speed_scale"]
        if out_mag > max_step and out_mag > 1e-9:
            raw_x *= max_step / out_mag
            raw_y *= max_step / out_mag

        prev_output_x = raw_x
        prev_output_y = raw_y
        cursor_speed_x = raw_x / max(1e-6, dt)
        cursor_speed_y = raw_y / max(1e-6, dt)
        if confidence > 0.0:
            state["error_x"] -= raw_x
            state["error_y"] -= raw_y

        row: dict[str, float | str | int] = {
            "profile": profile,
            "episode": episode,
            "step": step,
        }
        row.update(features)
        row.update(labels)

        # Populate the 5 new columns (40 total) with synthetic but realistic values
        # to match the current C++ PidGovernorInput / pidGovernorFeatures contract.
        box_aspect = float(state.get("target_width", 20.0)) / max(1e-6, float(state.get("target_height", 20.0)))
        # aim_point_error = raw observed error (what the command saw before any offset correction)
        aim_err_x = float(observed_x)
        aim_err_y = float(observed_y)
        # target_offset simulates a prior neural / advisory correction already active this frame.
        # In real runtime this comes from NeuralTargetingHead or other sources.
        # We occasionally inject a plausible non-zero offset so the model learns to react to it.
        offset_x = 0.0
        offset_y = 0.0
        if rng.random() < 0.28:
            # small fraction of the just-computed correction (or a damped random nudge)
            offset_x = float(raw_x) * rng.uniform(0.05, 0.55)
            offset_y = float(raw_y) * rng.uniform(0.05, 0.55)
            if rng.random() < 0.25:
                # occasional independent small offset (different source)
                offset_x += rng.gauss(0.0, max(0.8, abs(observed_x) * 0.12))
                offset_y += rng.gauss(0.0, max(0.8, abs(observed_y) * 0.12))

        row["target_offset_x"] = float(offset_x)
        row["target_offset_y"] = float(offset_y)
        row["aim_point_error_x"] = float(aim_err_x)
        row["aim_point_error_y"] = float(aim_err_y)
        row["box_aspect_ratio"] = float(np.clip(box_aspect, 0.25, 4.0))

        rows.append(row)

        prev_observed_x = observed_x
        prev_observed_y = observed_y
        prev_distance = distance

    return rows


def generate_dataset(pid: PidConfig, cfg: DatasetConfig) -> list[dict[str, float | str | int]]:
    rng = random.Random(cfg.seed)
    rows: list[dict[str, float | str | int]] = []
    for profile in PROFILE_NAMES:
        for episode in range(cfg.episodes_per_profile):
            rows.extend(_simulate_profile(profile, episode, rng, pid, cfg))
    return rows


def write_dataset(path: Path, rows: Iterable[dict[str, float | str | int]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["profile", "episode", "step", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_dataset(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | str]] = []
        for raw in reader:
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                if key in {"profile"}:
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
        return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic PID governor training dataset.")
    parser.add_argument("--config", default="x64/DML/config.ini", help="Runtime config.ini to read PID defaults from.")
    parser.add_argument("--output", default="training/data/pid_governor_dataset.csv", help="Output CSV path.")
    parser.add_argument("--episodes-per-profile", type=int, default=64)
    parser.add_argument("--steps-per-episode", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-speed-multiple", type=float, default=5.0)
    parser.add_argument("--min-target-size-px", type=float, default=6.0)
    parser.add_argument("--max-target-size-px", type=float, default=96.0)
    parser.add_argument("--detection-noise-px", type=float, default=0.35)
    parser.add_argument("--dt-jitter-fraction", type=float, default=0.10)
    parser.add_argument("--max-speed-px-s", type=float, default=0.0, help="Optional absolute synthetic target speed cap.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pid = load_pid_config(resolve_repo_path(args.config))
    cfg = DatasetConfig(
        episodes_per_profile=max(1, args.episodes_per_profile),
        steps_per_episode=max(8, args.steps_per_episode),
        seed=args.seed,
        max_speed_multiple=_clamp(args.max_speed_multiple, 1.0, 100.0),
        min_target_size_px=max(1.0, args.min_target_size_px),
        max_target_size_px=max(args.min_target_size_px, args.max_target_size_px),
        detection_noise_px=max(0.0, args.detection_noise_px),
        dt_jitter_fraction=_clamp(args.dt_jitter_fraction, 0.0, 0.50),
        max_target_speed_px_s=args.max_speed_px_s if args.max_speed_px_s > 0.0 else None,
    )
    rows = generate_dataset(pid, cfg)
    output = resolve_repo_path(args.output)
    write_dataset(output, rows)
    print(f"Wrote {len(rows)} samples to {output}")
    print(f"Profiles: {', '.join(PROFILE_NAMES)}")
    print(f"Current effective speed: {pid.nominal_speed_px_s:.2f} px/s")
    print(f"Max synthetic target speed: {cfg.speed_cap_px_s(pid):.2f} px/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
