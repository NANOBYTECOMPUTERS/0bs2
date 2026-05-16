import unittest
from pathlib import Path

from training.evaluate_pid_governor import resolve_repo_path as resolve_eval_path
from training.export_pid_governor_onnx import resolve_repo_path as resolve_export_path
from training.pid_governor.dataset import resolve_repo_path as resolve_dataset_path
from training.train_pid_governor import resolve_repo_path as resolve_train_path


REPO_ROOT = Path(__file__).resolve().parents[2]


class PidGovernorCliPathTests(unittest.TestCase):
    def test_training_script_defaults_resolve_from_repo_root(self):
        expected = REPO_ROOT / "training" / "data" / "pid_governor_dataset.csv"

        self.assertEqual(resolve_train_path("training/data/pid_governor_dataset.csv"), expected)
        self.assertEqual(resolve_eval_path("training/data/pid_governor_dataset.csv"), expected)
        self.assertEqual(resolve_dataset_path("training/data/pid_governor_dataset.csv"), expected)

    def test_export_script_defaults_resolve_from_repo_root(self):
        expected = REPO_ROOT / "training" / "models" / "pid_governor.pt"

        self.assertEqual(resolve_export_path("training/models/pid_governor.pt"), expected)

    def test_absolute_paths_are_preserved(self):
        absolute = REPO_ROOT / "custom" / "file.csv"

        self.assertEqual(resolve_train_path(str(absolute)), absolute)
        self.assertEqual(resolve_export_path(str(absolute)), absolute)
        self.assertEqual(resolve_eval_path(str(absolute)), absolute)
        self.assertEqual(resolve_dataset_path(str(absolute)), absolute)


if __name__ == "__main__":
    unittest.main()
