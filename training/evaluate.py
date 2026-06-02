#!/usr/bin/env python3
"""Evaluate temporal predictor checkpoints or exported ONNX models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.temporal_predictor.dataset import FEATURE_COLUMNS, load_temporal_npz, resolve_repo_path
from training.temporal_predictor.model import NormalizedTemporalPredictor, make_model


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required to evaluate .pt temporal predictor checkpoints.") from exc
    return torch


def predict_with_checkpoint(model_path: Path, history: np.ndarray, device_name: str) -> np.ndarray:
    torch = import_torch()
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    core = make_model(
        feature_dim=len(artifact["feature_columns"]),
        history_length=int(artifact["history_length"]),
        prediction_horizon=int(artifact["prediction_horizon"]),
        hidden_size=int(artifact["hidden_size"]),
        model_type=str(artifact["model_type"]),
    )
    core.load_state_dict(artifact["state_dict"])
    core.eval()
    wrapped = NormalizedTemporalPredictor(core, artifact["feature_mean"], artifact["feature_std"])
    wrapped.eval()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    wrapped.to(device)
    with torch.no_grad():
        tensor = torch.tensor(history, dtype=torch.float32, device=device)
        return wrapped(tensor).detach().cpu().numpy()


def predict_with_onnx(model_path: Path, history: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit("onnxruntime is required to evaluate exported ONNX temporal predictor models.") from exc

    providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    session = ort.InferenceSession(
        str(model_path),
        providers=[provider for provider in providers if provider in available],
    )
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: history.astype(np.float32, copy=False)})[0]
    return np.asarray(output, dtype=np.float32)


def compute_metrics(prediction: np.ndarray, target: np.ndarray, near_horizon_frames: int = 6) -> dict:
    error = np.linalg.norm(prediction - target, axis=-1)
    near_frames = max(1, min(int(near_horizon_frames), prediction.shape[1]))
    deltas = np.diff(prediction, axis=1)
    smoothness_score = float(np.var(deltas, axis=(0, 1)).sum()) if deltas.size else 0.0
    return {
        "ade": float(np.mean(error)),  # Average Displacement Error
        "near_ade": float(np.mean(error[:, :near_frames])),
        "fde": float(np.mean(error[:, -1])),  # Final Displacement Error
        "smoothness_score": smoothness_score,
        "samples": int(prediction.shape[0]),
        "prediction_horizon": int(prediction.shape[1]),
        "near_horizon_frames": int(near_frames),
    }


def predict_any_model(model_path: Path, history: np.ndarray, device_name: str) -> np.ndarray:
    if model_path.suffix.lower() == ".onnx":
        return predict_with_onnx(model_path, history)
    return predict_with_checkpoint(model_path, history, device_name)


def write_plots(history: np.ndarray, target: np.ndarray, prediction: np.ndarray, output_dir: Path, count: int) -> int:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping temporal predictor plots.")
        return 0

    written = 0
    for idx in range(min(max(0, count), history.shape[0])):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.plot(history[idx, :, 0], history[idx, :, 1], "o-", label="history", alpha=0.75)
        ax.plot(target[idx, :, 0], target[idx, :, 1], "o-", label="future ground truth", alpha=0.85)
        ax.plot(prediction[idx, :, 0], prediction[idx, :, 1], "o-", label="prediction", alpha=0.85)
        ax.set_title(f"Temporal prediction sample {idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = output_dir / f"trajectory_{idx:03d}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written += 1
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate temporal predictor trajectories.")
    parser.add_argument("--dataset", default="training/datasets/temporal_tracks.npz")
    parser.add_argument("--real-world-data", default=None, help="Optional held-out real-world temporal dataset.")
    parser.add_argument("--model", default="training/models/temporal_predictor.pt")
    parser.add_argument("--compare-model", default=None, help="Optional baseline model for before/after comparison.")
    parser.add_argument("--output-dir", default="training/models/temporal_eval")
    parser.add_argument("--samples-to-plot", type=int, default=8)
    parser.add_argument("--near-horizon-frames", type=int, default=6)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = load_temporal_npz(args.real_world_data or args.dataset)
    model_path = resolve_repo_path(args.model)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = predict_any_model(model_path, dataset.history, args.device)

    if prediction.shape != dataset.future.shape:
        raise SystemExit(
            f"Model output shape {prediction.shape} does not match target shape {dataset.future.shape}."
        )

    metrics = compute_metrics(prediction, dataset.future, args.near_horizon_frames)
    if args.compare_model:
        compare_path = resolve_repo_path(args.compare_model)
        compare_prediction = predict_any_model(compare_path, dataset.history, args.device)
        if compare_prediction.shape != dataset.future.shape:
            raise SystemExit(
                f"Compare model output shape {compare_prediction.shape} does not match target shape {dataset.future.shape}."
            )
        compare_metrics = compute_metrics(compare_prediction, dataset.future, args.near_horizon_frames)
        metrics["comparison"] = {
            "candidate_model": str(model_path),
            "compare_model": str(compare_path),
            "candidate": {key: metrics[key] for key in ("near_ade", "ade", "fde", "smoothness_score")},
            "baseline": {key: compare_metrics[key] for key in ("near_ade", "ade", "fde", "smoothness_score")},
            "delta_vs_baseline": {
                key: metrics[key] - compare_metrics[key]
                for key in ("near_ade", "ade", "fde", "smoothness_score")
            },
        }

    plot_count = write_plots(dataset.history, dataset.future, prediction, output_dir, args.samples_to_plot)
    metrics["plots"] = int(plot_count)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Temporal predictor evaluation: "
        f"near_ADE={metrics['near_ade']:.4f} ADE={metrics['ade']:.4f} FDE={metrics['fde']:.4f} "
        f"smoothness={metrics['smoothness_score']:.4f}"
    )
    if "comparison" in metrics:
        delta = metrics["comparison"]["delta_vs_baseline"]
        print(
            "Comparison delta vs baseline: "
            f"near_ADE={delta['near_ade']:.4f} ADE={delta['ade']:.4f} "
            f"FDE={delta['fde']:.4f} smoothness={delta['smoothness_score']:.4f}"
        )
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
