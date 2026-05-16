import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "overlay" / "draw_mouse.cpp").exists()
)


class PidGovernorGuiContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_mouse_gui_exposes_governor_controls_without_model_path(self):
        overlay_cpp = self.read("overlay/draw_mouse.cpp")

        self.assertIn('ImGui::Checkbox("Enable PID governor", &config.pid_governor_enabled)', overlay_cpp)
        self.assertIn(
            'ImGui::SliderFloat("Governor blend", &config.pid_governor_blend, 0.0f, 1.0f, "%.2f")',
            overlay_cpp,
        )
        self.assertIn(
            'ImGui::SliderFloat("Governor max speed multiple", &config.pid_governor_max_speed_multiple, 1.0f, 5.0f, "%.2f")',
            overlay_cpp,
        )
        self.assertNotIn("pid_governor_model_path", overlay_cpp)
        self.assertNotIn("Governor model", overlay_cpp)

    def test_governor_control_changes_refresh_mouse_thread(self):
        overlay_cpp = self.read("overlay/draw_mouse.cpp")

        self.assertIn("prev_pid_governor_enabled != config.pid_governor_enabled", overlay_cpp)
        self.assertIn("prev_pid_governor_blend != config.pid_governor_blend", overlay_cpp)
        self.assertIn(
            "prev_pid_governor_max_speed_multiple != config.pid_governor_max_speed_multiple",
            overlay_cpp,
        )
        self.assertIn("prev_pid_governor_enabled = config.pid_governor_enabled", overlay_cpp)
        self.assertIn("prev_pid_governor_blend = config.pid_governor_blend", overlay_cpp)
        self.assertIn(
            "prev_pid_governor_max_speed_multiple = config.pid_governor_max_speed_multiple",
            overlay_cpp,
        )

    def test_settings_reference_tracks_governor_gui_visibility(self):
        generator = self.read("docs/settings-reference/generate_settings_reference.py")

        self.assertIn(
            'row("Mouse", "Pure PID Movement", "Enable PID governor", "pid_governor_enabled"',
            generator,
        )
        self.assertIn(
            'row("Mouse", "Pure PID Movement", "Governor blend", "pid_governor_blend"',
            generator,
        )
        self.assertIn(
            'row("Mouse", "Pure PID Movement", "Governor max speed multiple", "pid_governor_max_speed_multiple"',
            generator,
        )
        self.assertIn(
            'config_only("PID governor", "pid_governor_model_path"',
            generator,
        )
        self.assertNotIn(
            'config_only("PID governor", "pid_governor_enabled"',
            generator,
        )
        self.assertNotIn(
            'config_only("PID governor", "pid_governor_blend"',
            generator,
        )
        self.assertNotIn(
            'config_only("PID governor", "pid_governor_max_speed_multiple"',
            generator,
        )


if __name__ == "__main__":
    unittest.main()
