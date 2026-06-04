#!/usr/bin/env python3
"""Evaluate NeuralTargetingHead checkpoints or exported ONNX models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.neural_targeting.dataset import load_neural_targeting_npz, resolve_repo_path
from training.neural_targeting.model import make_model


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required to evaluate .pt neural targeting checkpoints.") from exc
    return torch


def predict_with_checkpoint(model_path: Path, features: np.ndarray, device_name: str) -> np.ndarray:
    torch = import_torch()
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    model = make_model(
        feature_dim=int(artifact["feature_dim"]),
        hidden_size=int(artifact["hidden_size"]),
        max_refinement_px=float(artifact["max_refinement_px"]),
        dropout=float(artifact.get("dropout", 0.0)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy()


def predict_with_onnx(model_path: Path, features: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit("onnxruntime is required to evaluate exported ONNX neural targeting models.") from exc

    providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    session = ort.InferenceSession(
        str(model_path),
        providers=[provider for provider in providers if provider in available],
    )
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: features.astype(np.float32, copy=False)})[0]
    return np.asarray(output, dtype=np.float32)


def compute_metrics(prediction: np.ndarray, refinement: np.ndarray, confidence: np.ndarray) -> dict:
    pred_refinement = prediction[:, :2]
    pred_confidence = np.clip(prediction[:, 2:3], 0.0, 1.0)
    error = np.linalg.norm(pred_refinement - refinement, axis=1)
    target_mag = np.linalg.norm(refinement, axis=1)
    pred_mag = np.linalg.norm(pred_refinement, axis=1)
    low_conf_mask = confidence[:, 0] < 0.35
    overconfident_bad = np.logical_and(low_conf_mask, pred_confidence[:, 0] > 0.65)
    return {
        "refinement_mae": float(np.mean(np.abs(pred_refinement - refinement))),
        "refinement_rmse": float(np.sqrt(np.mean((pred_refinement - refinement) ** 2))),
        "refinement_vector_error": float(np.mean(error)),
        "confidence_mae": float(np.mean(np.abs(pred_confidence - confidence))),
        "mean_target_magnitude": float(np.mean(target_mag)),
        "mean_predicted_magnitude": float(np.mean(pred_mag)),
        "bad_overconfidence_rate": float(np.mean(overconfident_bad)) if overconfident_bad.size else 0.0,
        "samples": int(prediction.shape[0]),
    }


def predict_any_model(model_path: Path, features: np.ndarray, device_name: str) -> np.ndarray:
    if model_path.suffix.lower() == ".onnx":
        return predict_with_onnx(model_path, features)
    return predict_with_checkpoint(model_path, features, device_name)


def write_plots(features: np.ndarray, target: np.ndarray, prediction: np.ndarray, output_dir: Path, count: int) -> int:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping neural targeting plots.")
        return 0

    written = 0
    for idx in range(min(max(0, count), features.shape[0])):
        center = np.array([features[idx, 0] * 640.0, features[idx, 1] * 640.0], dtype=np.float32)
        target_point = center + target[idx]
        predicted_point = center + prediction[idx, :2]
        future = []
        cursor = 10
        while cursor + 1 < features.shape[1]:
            future.append(
                [
                    center[0] + features[idx, cursor] * 640.0,
                    center[1] + features[idx, cursor + 1] * 640.0,
                ]
            )
            cursor += 2
        future = np.asarray(future, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.plot(center[0], center[1], "ko", label="current")
        if future.size:
            ax.plot(future[:, 0], future[:, 1], "o-", label="predicted trajectory", alpha=0.65)
        ax.plot(target_point[0], target_point[1], "go", label="ideal refinement")
        ax.plot(predicted_point[0], predicted_point[1], "ro", label="model refinement")
        ax.set_title(f"Neural targeting sample {idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = output_dir / f"targeting_{idx:03d}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written += 1
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate neural targeting refinement predictions.")
    parser.add_argument("--dataset", default="training/datasets/neural_targeting_tracks.npz")
    parser.add_argument("--real-world-data", default=None, help="Optional held-out real-world neural targeting dataset.")
    parser.add_argument("--model", default="training/models/neural_targeting_head.pt")
    parser.add_argument("--compare-model", default=None, help="Optional baseline model for before/after comparison.")
    parser.add_argument("--output-dir", default="training/models/neural_targeting_eval")
    parser.add_argument("--samples-to-plot", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = load_neural_targeting_npz(args.real_world_data or args.dataset)
    model_path = resolve_repo_path(args.model)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = predict_any_model(model_path, dataset.features, args.device)

    if prediction.shape != (dataset.features.shape[0], 3):
        raise SystemExit(f"Model output shape {prediction.shape} does not match expected {(dataset.features.shape[0], 3)}.")

    metrics = compute_metrics(prediction, dataset.refinement, dataset.confidence)
    if args.compare_model:
        compare_path = resolve_repo_path(args.compare_model)
        compare_prediction = predict_any_model(compare_path, dataset.features, args.device)
        if compare_prediction.shape != (dataset.features.shape[0], 3):
            raise SystemExit(
                f"Compare model output shape {compare_prediction.shape} does not match expected {(dataset.features.shape[0], 3)}."
            )
        compare_metrics = compute_metrics(compare_prediction, dataset.refinement, dataset.confidence)
        comparison_keys = (
            "refinement_mae",
            "refinement_rmse",
            "refinement_vector_error",
            "confidence_mae",
            "bad_overconfidence_rate",
        )
        metrics["comparison"] = {
            "candidate_model": str(model_path),
            "compare_model": str(compare_path),
            "candidate": {key: metrics[key] for key in comparison_keys},
            "baseline": {key: compare_metrics[key] for key in comparison_keys},
            "delta_vs_baseline": {key: metrics[key] - compare_metrics[key] for key in comparison_keys},
        }

    plot_count = write_plots(dataset.features, dataset.refinement, prediction, output_dir, args.samples_to_plot)
    metrics["plots"] = int(plot_count)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Neural targeting evaluation: "
        f"refinement_MAE={metrics['refinement_mae']:.4f} "
        f"refinement_RMSE={metrics['refinement_rmse']:.4f} "
        f"confidence_MAE={metrics['confidence_mae']:.4f} "
        f"bad_overconfidence={metrics['bad_overconfidence_rate']:.4f}"
    )
    if "comparison" in metrics:
        delta = metrics["comparison"]["delta_vs_baseline"]
        print(
            "Comparison delta vs baseline: "
            f"refinement_MAE={delta['refinement_mae']:.4f} "
            f"refinement_RMSE={delta['refinement_rmse']:.4f} "
            f"confidence_MAE={delta['confidence_mae']:.4f}"
        )
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
