import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class TemporalTrainerContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_training_files_and_contract_tokens_exist(self):
        expected_files = [
            "training/data_gen/generate_track_data.py",
            "training/temporal_predictor/dataset.py",
            "training/temporal_predictor/model.py",
            "training/train_temporal.py",
            "training/evaluate.py",
        ]
        for relative_path in expected_files:
            self.assertTrue((REPO_ROOT / relative_path).exists(), relative_path)

        generator = self.read("training/data_gen/generate_track_data.py")
        train = self.read("training/train_temporal.py")
        evaluate = self.read("training/evaluate.py")
        model = self.read("training/temporal_predictor/model.py")

        for token in (
            "FEATURE_COLUMNS",
            "TrackDataConfig",
            "constant_velocity",
            "acceleration",
            "curve",
            "fast_maneuver",
            "partial_occlusion",
            "camera_shake",
            "np.savez_compressed",
        ):
            self.assertIn(token, generator)

        for token in (
            "TemporalPredictorNet",
            "torch.nn.GRU",
            "TransformerEncoder",
            "NormalizedTemporalPredictor",
            "future_xy",
        ):
            self.assertIn(token, model)

        self.assertIn("hidden_size must be divisible by 4 when model_type='transformer'", model)
        self.assertIn("temporal_history length", model)
        self.assertIn("exceeds configured history_length", model)

        for token in (
            "AdamW",
            "ReduceLROnPlateau",
            "early stopping",
            "smoothness_loss",
            "velocity_consistency_loss",
            "near_horizon_weight",
            "latency_focus_loss",
            "latency_focus_weight",
            "val_latency_ade",
            "weighted_position_loss",
            "torch.onnx.export",
            "models/temporal_predictor.onnx",
            "dynamic_axes",
        ):
            self.assertIn(token, train)

        for token in (
            "Average Displacement Error",
            "latency_ade",
            "Final Displacement Error",
            "smoothness_score",
            "matplotlib",
            "onnxruntime",
        ):
            self.assertIn(token, evaluate)

    def test_generator_writes_npz_with_runtime_feature_contract(self):
        from training.data_gen.generate_track_data import (
            FEATURE_COLUMNS,
            TrackDataConfig,
            generate_dataset,
            save_dataset,
        )

        cfg = TrackDataConfig(samples=24, history_length=6, prediction_horizon=4, seed=17)
        history, future, profiles, metadata = generate_dataset(cfg)

        self.assertEqual(
            FEATURE_COLUMNS,
            ["x", "y", "w", "h", "vx", "vy", "box_scale_vel", "confidence"],
        )
        self.assertEqual(history.shape, (24, 6, 8))
        self.assertEqual(future.shape, (24, 4, 2))
        self.assertEqual(len(profiles), 24)
        self.assertEqual(metadata["history_length"], 6)
        self.assertEqual(metadata["prediction_horizon"], 4)
        self.assertGreater(len(set(profiles.tolist())), 2)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tracks.npz"
            save_dataset(output, cfg)
            self.assertTrue(output.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for loss weighting tests.")
    def test_near_horizon_weighted_loss_prioritizes_first_frames(self):
        import torch

        from training.train_temporal import build_horizon_weights, weighted_position_loss

        weights = build_horizon_weights(torch, horizon=8, near_frames=3, near_weight=2.5, device="cpu")
        self.assertGreater(float(weights[0]), float(weights[-1]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

        target = torch.zeros(1, 8, 2)
        early_error = target.clone()
        late_error = target.clone()
        early_error[:, 0, :] = 4.0
        late_error[:, -1, :] = 4.0

        early_loss = weighted_position_loss(torch, early_error, target, weights)
        late_loss = weighted_position_loss(torch, late_error, target, weights)
        self.assertGreater(float(early_loss), float(late_loss))

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("numpy")
        and importlib.util.find_spec("onnx"),
        "PyTorch, NumPy, and ONNX are required for the temporal trainer smoke test.",
    )
    def test_train_script_auto_generates_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "missing_tracks.npz"
            model_path = tmp_path / "temporal_predictor.pt"
            onnx_path = tmp_path / "temporal_predictor.onnx"
            metadata_path = tmp_path / "temporal_predictor.json"

            train_cmd = [
                sys.executable,
                str(REPO_ROOT / "training" / "train_temporal.py"),
                "--dataset",
                str(dataset_path),
                "--output",
                str(model_path),
                "--onnx-output",
                str(onnx_path),
                "--metadata",
                str(metadata_path),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--hidden-size",
                "16",
                "--history-length",
                "6",
                "--prediction-horizon",
                "4",
                "--auto-generate-samples",
                "24",
                "--patience",
                "1",
                "--no-tensorboard",
            ]
            result = subprocess.run(
                train_cmd,
                cwd=REPO_ROOT / "training",
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(dataset_path.exists())
            self.assertTrue(model_path.exists())
            self.assertIn("Auto-generated missing temporal dataset", result.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("numpy")
        and importlib.util.find_spec("onnx"),
        "PyTorch, NumPy, and ONNX are required for the temporal trainer smoke test.",
    )
    def test_tiny_training_run_exports_onnx_and_evaluates(self):
        from training.data_gen.generate_track_data import TrackDataConfig, save_dataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "tracks.npz"
            model_path = tmp_path / "temporal_predictor.pt"
            onnx_path = tmp_path / "temporal_predictor.onnx"
            metadata_path = tmp_path / "temporal_predictor.json"
            eval_dir = tmp_path / "eval"

            save_dataset(
                dataset_path,
                TrackDataConfig(samples=48, history_length=6, prediction_horizon=4, seed=31),
            )

            train_cmd = [
                sys.executable,
                str(REPO_ROOT / "training" / "train_temporal.py"),
                "--dataset",
                str(dataset_path),
                "--output",
                str(model_path),
                "--onnx-output",
                str(onnx_path),
                "--metadata",
                str(metadata_path),
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--hidden-size",
                "16",
                "--history-length",
                "6",
                "--prediction-horizon",
                "4",
                "--patience",
                "1",
                "--no-tensorboard",
            ]
            subprocess.run(train_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            self.assertTrue(model_path.exists())
            self.assertTrue(onnx_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_shape"], [1, 6, 8])
            self.assertEqual(metadata["output_shape"], [1, 4, 2])

            eval_cmd = [
                sys.executable,
                str(REPO_ROOT / "training" / "evaluate.py"),
                "--dataset",
                str(dataset_path),
                "--model",
                str(model_path),
                "--output-dir",
                str(eval_dir),
                "--samples-to-plot",
                "1",
            ]
            subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("ade", metrics)
            self.assertIn("near_ade", metrics)
            self.assertIn("fde", metrics)
            self.assertIn("smoothness_score", metrics)
            self.assertGreaterEqual(metrics["ade"], 0.0)

            if importlib.util.find_spec("onnxruntime"):
                eval_onnx_dir = tmp_path / "eval_onnx"
                eval_onnx_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "training" / "evaluate.py"),
                    "--dataset",
                    str(dataset_path),
                    "--model",
                    str(onnx_path),
                    "--output-dir",
                    str(eval_onnx_dir),
                    "--samples-to-plot",
                    "0",
                ]
                subprocess.run(eval_onnx_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
                onnx_metrics = json.loads((eval_onnx_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertIn("ade", onnx_metrics)
                self.assertEqual(onnx_metrics["prediction_horizon"], 4)


if __name__ == "__main__":
    unittest.main()
