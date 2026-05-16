import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "train_pid_governor_auto.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_pid_governor_auto", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoTrainPidGovernorTests(unittest.TestCase):
    def test_script_exists_at_repo_root(self):
        self.assertTrue(SCRIPT_PATH.exists())

    def test_reads_pid_values_from_config(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.ini"
            config.write_text(
                "\n".join(
                    [
                        "pid_actuator_hz = 140",
                        "pid_kp = 0.0250",
                        "pid_ki = 0.0020",
                        "pid_kd = 0.0009",
                        "pid_max_pixel_step = 1.000",
                        "pid_output_scale = 0.100",
                        "pid_governor_max_speed_multiple = 4.250",
                        "pid_governor_model_path = training/models/pid_governor.onnx",
                    ]
                ),
                encoding="utf-8",
            )

            values = module.read_scalar_config(config)

        self.assertEqual(values["pid_actuator_hz"], "140")
        self.assertEqual(values["pid_kp"], "0.0250")
        self.assertEqual(module.config_float(values, "pid_governor_max_speed_multiple", 5.0), 4.25)

    def test_builds_generate_train_export_commands_from_root_paths(self):
        module = load_script_module()
        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "x64/DML/config.ini",
                "--dataset",
                "training/data/pid_governor_dataset.csv",
                "--models-dir",
                "training/models",
                "--epochs",
                "3",
                "--max-speed-multiple",
                "4.5",
            ]
        )

        commands = module.build_commands(args, {"pid_governor_max_speed_multiple": "3.5"})
        flat = [" ".join(str(part) for part in command) for command in commands]

        self.assertEqual(len(commands), 3)
        self.assertIn(str(REPO_ROOT / "training" / "generate_pid_dataset.py"), flat[0])
        self.assertIn(str(REPO_ROOT / "x64" / "DML" / "config.ini"), flat[0])
        self.assertIn("--max-speed-multiple 4.5", flat[0])
        self.assertIn(str(REPO_ROOT / "training" / "train_pid_governor.py"), flat[1])
        self.assertIn("--epochs 3", flat[1])
        self.assertIn(str(REPO_ROOT / "training" / "export_pid_governor_onnx.py"), flat[2])

    def test_copy_targets_include_dml_root_and_config_relative_path(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dml_dir = root / "x64" / "DML"
            dml_dir.mkdir(parents=True)
            onnx = root / "training" / "models" / "pid_governor.onnx"
            meta = root / "training" / "models" / "pid_governor_onnx.json"
            onnx.parent.mkdir(parents=True)
            onnx.write_bytes(b"onnx")
            meta.write_text("{}", encoding="utf-8")

            copied = module.copy_governor_outputs(
                onnx,
                meta,
                dml_dir,
                {"pid_governor_model_path": "training/models/pid_governor.onnx"},
            )

            copied_paths = {path.relative_to(root).as_posix() for path in copied}

        self.assertIn("x64/DML/pid_governor.onnx", copied_paths)
        self.assertIn("x64/DML/pid_governor_onnx.json", copied_paths)
        self.assertIn("x64/DML/training/models/pid_governor.onnx", copied_paths)
        self.assertIn("x64/DML/training/models/pid_governor_onnx.json", copied_paths)


if __name__ == "__main__":
    unittest.main()
