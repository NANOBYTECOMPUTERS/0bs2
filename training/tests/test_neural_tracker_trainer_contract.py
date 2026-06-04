import unittest
from collections import defaultdict
from pathlib import Path

from training.neural_tracker.dataset import GROUP_COLUMN, LABEL_COLUMN, NeuralTrackerDatasetConfig, generate_synthetic_dataset


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class NeuralTrackerTrainerContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_synthetic_dataset_emits_grouped_hard_negatives(self):
        rows = generate_synthetic_dataset(NeuralTrackerDatasetConfig(samples=80, seed=9))
        self.assertTrue(all(GROUP_COLUMN in row for row in rows))

        grouped = defaultdict(list)
        for row in rows:
            grouped[row[GROUP_COLUMN]].append(float(row[LABEL_COLUMN]))

        paired_groups = [labels for labels in grouped.values() if any(label >= 0.5 for label in labels) and any(label < 0.5 for label in labels)]
        self.assertGreater(len(paired_groups), 0)
        self.assertTrue(any(str(row.get("source")) == "synthetic_group_hard_negative" for row in rows))
        self.assertTrue(any(float(row.get("class_compatible", 1.0)) == 0.5 for row in rows))

    def test_trainer_uses_group_split_train_only_teacher_ranking_and_calibration(self):
        source = self.read("training/train_neural_tracker.py")

        self.assertIn("split_indices_by_group", source)
        self.assertIn("group_key_for_row", source)
        self.assertIn("train_rows = [rows[index] for index in train_indices]", source)
        self.assertIn("try_train_gbm_teacher(train_rows, rows", source)
        self.assertIn("pairwise_ranking_loss", source)
        self.assertIn("--ranking-loss-weight", source)
        self.assertIn("fit_temperature", source)
        self.assertIn("calibration_temperature", source)
        self.assertIn("--real-world-data", source)

    def test_export_applies_saved_calibration_temperature(self):
        source = self.read("training/export_neural_tracker_onnx.py")

        self.assertIn("calibration_temperature", source)
        self.assertIn("torch.logit(score)", source)
        self.assertIn("torch.sigmoid(logit / self.temperature)", source)

    def test_evaluator_reports_grouped_and_calibration_metrics(self):
        source = self.read("training/evaluate_neural_tracker.py")

        for token in (
            "roc_auc",
            "pr_auc",
            "group_top1_accuracy",
            "calibration_temperature",
            "--compare-model",
            "--real-world-data",
        ):
            self.assertIn(token, source)

    def test_runtime_passes_soft_class_compatibility_to_neural_features(self):
        source = self.read("mouse/BoxTarget.cpp")

        self.assertIn("classCompatibilityScore", source)
        self.assertIn("classCompatibilityScore = 0.5", source)
        self.assertIn("static_cast<float>(classCompatibilityScore)", source)
        self.assertNotIn("headingAlignment,\n                    true,\n                    relaxedForLocked", source)


if __name__ == "__main__":
    unittest.main()
