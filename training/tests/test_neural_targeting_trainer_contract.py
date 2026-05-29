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


class NeuralTargetingTrainerContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_training_files_and_runtime_contract_tokens_exist(self):
        expected_files = [
            "training/data_gen/generate_targeting_data.py",
            "training/neural_targeting/__init__.py",
            "training/neural_targeting/dataset.py",
            "training/neural_targeting/model.py",
            "training/train_neural_targeting.py",
            "training/evaluate_neural_targeting.py",
        ]
        for relative_path in expected_files:
            self.assertTrue((REPO_ROOT / relative_path).exists(), relative_path)

        generator = self.read("training/data_gen/generate_targeting_data.py")
        dataset = self.read("training/neural_targeting/dataset.py")
        model = self.read("training/neural_targeting/model.py")
        train = self.read("training/train_neural_targeting.py")

        for token in (
            "TargetingDataConfig",
            "generate_targeting_dataset",
            "bad_prediction_probability",
            "near_lock_damping",
            "np.savez_compressed",
        ):
            self.assertIn(token, generator)

        for token in (
            "BASE_FEATURE_COLUMNS",
            "future_dx_",
            "future_dy_",
            "load_neural_targeting_npz",
            "refinement",
            "confidence",
        ):
            self.assertIn(token, dataset)

        for token in (
            "NeuralTargetingNet",
            "torch.nn.Sequential",
            "LayerNorm",
            "torch.tanh",
            "torch.sigmoid",
            "targeting_output",
        ):
            self.assertIn(token, model)

        for token in (
            "models/neural_targeting_head.onnx",
            "targeting_features",
            "targeting_output",
            "refinement_loss",
            "confidence_calibration_loss",
            "near_lock_penalty_loss",
            "output_magnitude_loss",
            "torch.onnx.export",
            "dynamic_axes",
        ):
            self.assertIn(token, train)

    def test_generator_writes_npz_matching_cpp_feature_contract(self):
        from training.data_gen.generate_targeting_data import TargetingDataConfig, generate_targeting_dataset, save_dataset
        from training.neural_targeting.dataset import BASE_FEATURE_COLUMNS, load_neural_targeting_npz

        cfg = TargetingDataConfig(samples=32, prediction_horizon=5, seed=7)
        features, refinement, confidence, metadata = generate_targeting_dataset(cfg)
        expected_feature_dim = len(BASE_FEATURE_COLUMNS) + cfg.prediction_horizon * 2

        self.assertEqual(features.shape, (32, expected_feature_dim))
        self.assertEqual(refinement.shape, (32, 2))
        self.assertEqual(confidence.shape, (32, 1))
        self.assertEqual(metadata["prediction_horizon"], 5)
        self.assertEqual(metadata["input_feature_count"], expected_feature_dim)
        self.assertLessEqual(float(abs(refinement).max()), cfg.max_refinement_px + 1e-5)
        self.assertGreaterEqual(float(confidence.min()), 0.0)
        self.assertLessEqual(float(confidence.max()), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "targeting.npz"
            save_dataset(output, cfg)
            loaded = load_neural_targeting_npz(output)
            self.assertEqual(loaded.features.shape, features.shape)
            self.assertEqual(loaded.refinement.shape, refinement.shape)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for model shape tests.")
    def test_model_outputs_bounded_offsets_and_confidence(self):
        import torch

        from training.neural_targeting.model import NeuralTargetingNet

        model = NeuralTargetingNet(feature_dim=20, hidden_size=16, max_refinement_px=35.0)
        out = model(torch.zeros(4, 20))
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertLessEqual(float(out[:, :2].abs().max()), 35.0 + 1e-5)
        self.assertGreaterEqual(float(out[:, 2].min()), 0.0)
        self.assertLessEqual(float(out[:, 2].max()), 1.0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("numpy")
        and importlib.util.find_spec("onnx"),
        "PyTorch, NumPy, and ONNX are required for the neural targeting trainer smoke test.",
    )
    def test_tiny_training_run_exports_runtime_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "targeting_missing.npz"
            model_path = tmp_path / "neural_targeting_head.pt"
            onnx_path = tmp_path / "neural_targeting_head.onnx"
            metadata_path = tmp_path / "neural_targeting_head.json"

            cmd = [
                sys.executable,
                str(REPO_ROOT / "training" / "train_neural_targeting.py"),
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
                "--prediction-horizon",
                "4",
                "--auto-generate-samples",
                "32",
                "--no-tensorboard",
            ]
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)

            self.assertIn("Auto-generated missing neural targeting dataset", result.stdout)
            self.assertTrue(dataset_path.exists())
            self.assertTrue(model_path.exists())
            self.assertTrue(onnx_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_name"], "targeting_features")
            self.assertEqual(metadata["output_name"], "targeting_output")
            self.assertEqual(metadata["output_shape"], [1, 3])
            self.assertEqual(metadata["prediction_horizon"], 4)


if __name__ == "__main__":
    unittest.main()
