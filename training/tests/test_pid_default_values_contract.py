import tempfile
import unittest
from pathlib import Path

from training.pid_governor.dataset import PidConfig, load_pid_config


REPO_ROOT = Path(__file__).resolve().parents[4]


class PidDefaultValuesContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_source_config_defaults_match_working_dml_pid_values(self):
        config_cpp = self.read("config/config.cpp")
        overlay_cpp = self.read("overlay/draw_mouse.cpp")
        mouse_h = self.read("mouse/PidMouseController.h")
        mouse_cpp = self.read("mouse/PidMouseController.cpp")

        for source in (config_cpp, overlay_cpp):
            self.assertIn("pid_actuator_hz = 1000", source)
            self.assertIn("pid_kp = 0.0085f", source)
            self.assertIn("pid_ki = 0.0003f", source)
            self.assertIn("pid_kd = 0.0001f", source)
            self.assertIn("pid_deadzone_px = 0.0f", source)
            self.assertIn("pid_max_pixel_step = 0.80f", source)
            self.assertIn("pid_output_scale = 0.10f", source)

        self.assertIn('input_method = "RAZER"', config_cpp)
        self.assertIn("double kp = 0.0085", mouse_h)
        self.assertIn("double ki = 0.0003", mouse_h)
        self.assertIn("double kd = 0.0001", mouse_h)
        self.assertIn("int actuatorHz = 1000", mouse_h)
        self.assertIn("0.0085", mouse_cpp)
        self.assertIn("0.0003", mouse_cpp)
        self.assertIn("0.0001", mouse_cpp)

    def test_project_actuator_hz_cap_is_raised_to_2000(self):
        config_cpp = self.read("config/config.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")
        pid_cpp = self.read("mouse/PidMouseController.cpp")
        overlay_cpp = self.read("overlay/draw_mouse.cpp")

        self.assertIn("pid_actuator_hz > 2000", config_cpp)
        self.assertIn("30.0, 2000.0", mouse_cpp)
        self.assertIn("30.0, 2000.0", pid_cpp)
        self.assertIn('ImGui::SliderInt("Actuator Hz", &config.pid_actuator_hz, 30, 2000)', overlay_cpp)

    def test_dataset_defaults_match_working_dml_dataset_values(self):
        default_pid = PidConfig()
        self.assertEqual(default_pid.actuator_hz, 2000)
        self.assertAlmostEqual(default_pid.kp, 0.0200)
        self.assertAlmostEqual(default_pid.ki, 0.0003)
        self.assertAlmostEqual(default_pid.kd, 0.0001)
        self.assertAlmostEqual(default_pid.max_pixel_step, 0.80)
        self.assertAlmostEqual(default_pid.output_scale, 0.10)

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.ini"
            loaded = load_pid_config(missing)

        self.assertEqual(loaded.actuator_hz, 2000)
        self.assertAlmostEqual(loaded.kp, 0.0200)

        dataset_py = self.read("x64/DML/training/pid_governor/dataset.py")
        self.assertIn("target_size * 0.085", dataset_py)


if __name__ == "__main__":
    unittest.main()
