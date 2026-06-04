import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class PidGovernorTrainerContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_lightgbm_teacher_trains_one_regressor_per_output(self):
        source = self.read("training/train_pid_governor.py")

        self.assertIn("for output_index, label_column in enumerate(label_columns):", source)
        self.assertIn("label=y_matrix[:, output_index]", source)
        self.assertIn("np.stack(soft_columns, axis=1)", source)
        self.assertIn("np.clip(prediction, 0.0, 1.0)", source)
        self.assertNotIn("label=Y", source)


if __name__ == "__main__":
    unittest.main()
