import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "overlay" / "draw_mouse.cpp").exists()
)


class ConfigGuiControlContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_overlay_exposes_save_and_load_config_buttons(self):
        draw_overlay = self.read("overlay/draw_overlay.cpp")
        config_dirty_h = self.read("overlay/config_dirty.h")
        config_dirty_cpp = self.read("overlay/config_dirty.cpp")

        self.assertIn('ImGui::Button("Save Config"', draw_overlay)
        self.assertIn('ImGui::Button("Load Config"', draw_overlay)
        self.assertIn("SaveRuntimeConfig", draw_overlay)
        self.assertIn("LoadRuntimeConfigMerge", draw_overlay)
        self.assertIn("RefreshRuntimeAfterConfigLoad", self.read("0BS_box_2.cpp"))
        self.assertIn("globalMouseThread->updateConfig", self.read("0BS_box_2.cpp"))
        self.assertIn("OverlayConfig_SaveNow", config_dirty_h)
        self.assertIn("OverlayConfig_ClearDirty", config_dirty_cpp)

    def test_state_estimator_controls_are_live_in_mouse_gui(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for key in (
            "runtime_latency_sweep_enabled",
            "kalman_velocity_seed_enabled",
            "kalman_acquisition_frames",
            "kalman_process_noise_position",
            "kalman_process_noise_velocity",
            "kalman_measurement_noise",
        ):
            self.assertIn(key, config_h)
            self.assertIn(key, config_cpp)
            self.assertIn(key, draw_mouse)

        self.assertIn('OverlayUI::BeginSection("State Estimator"', draw_mouse)
        self.assertIn('ImGui::Checkbox("Runtime latency sweep"', draw_mouse)
        self.assertIn('ImGui::Checkbox("Seed velocity on acquire"', draw_mouse)
        self.assertIn('ImGui::SliderFloat("Process noise position"', draw_mouse)
        self.assertIn('ImGui::SliderFloat("Process noise velocity"', draw_mouse)
        self.assertIn('ImGui::SliderFloat("Measurement noise"', draw_mouse)
        self.assertIn("settings.runtimeLatencySweepEnabled = config.runtime_latency_sweep_enabled", mouse_cpp)
        self.assertIn("settings.velocitySeedEnabled = config.kalman_velocity_seed_enabled", mouse_cpp)
        self.assertIn("settings.acquisitionFrames = config.kalman_acquisition_frames", mouse_cpp)

    def test_pid_enhancement_controls_are_gated_and_documented(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        pid_h = self.read("mouse/PidMouseController.h")
        pid_cpp = self.read("mouse/PidMouseController.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        generator = self.read("docs/settings-reference/generate_settings_reference.py")

        for key in (
            "pid_feed_forward_frame_lookahead",
            "pid_conditional_integration_enabled",
            "pid_conditional_integration_error_px",
            "pid_adaptive_output_scaling_enabled",
            "pid_adaptive_output_error_scale",
            "pid_derivative_smoothing_multiplier",
            "pid_perspective_fov_mapping_enabled",
        ):
            self.assertIn(key, config_h)
            self.assertIn(key, config_cpp)
            self.assertIn(key, draw_mouse)
            self.assertIn(key, generator)

        for field in (
            "runtimeLatencySweepEnabled",
            "feedForwardFrameLookahead",
            "conditionalIntegrationEnabled",
            "conditionalIntegrationErrorPx",
            "adaptiveOutputScalingEnabled",
            "adaptiveOutputErrorScale",
            "derivativeSmoothingMultiplier",
            "perspectiveFovMappingEnabled",
            "projectionWidthPx",
            "fovXDeg",
            "angularDxDeg",
            "angularOutputActive",
        ):
            self.assertIn(field, pid_h)
            self.assertIn(field, pid_cpp)

        self.assertIn("settings.runtimeLatencySweepEnabled && settings.conditionalIntegrationEnabled", pid_cpp)
        self.assertIn("settings.runtimeLatencySweepEnabled && settings.adaptiveOutputScalingEnabled", pid_cpp)
        self.assertIn("settings.runtimeLatencySweepEnabled && settings.perspectiveFovMappingEnabled", pid_cpp)
        self.assertIn("settings.runtimeLatencySweepEnabled ? settings.feedForwardGain", pid_cpp)
        self.assertIn("settings.runtimeLatencySweepEnabled ? settings.derivativeSmoothingMultiplier", pid_cpp)
        self.assertIn("settings.perspectiveFovMappingEnabled = config.pid_perspective_fov_mapping_enabled", mouse_cpp := self.read("mouse/mouse.cpp"))
        self.assertIn("settings.projectionWidthPx = screen_width", mouse_cpp)
        self.assertIn("command.angularOutputActive", mouse_cpp)
        self.assertIn('ImGui::SliderFloat("Feed-forward gain", &config.pid_feed_forward_gain, 0.0f, 4.0f', draw_mouse)
        self.assertIn('ImGui::Checkbox("Perspective FOV PID"', draw_mouse)
        self.assertIn('ImGui::SliderInt("Feed-forward frame lookahead"', draw_mouse)
        self.assertIn('ImGui::Checkbox("Conditional integration"', draw_mouse)
        self.assertIn('ImGui::Checkbox("Adaptive output scaling"', draw_mouse)
        self.assertIn('ImGui::SliderFloat("Derivative smoothing multiplier"', draw_mouse)

    def test_target_offset_controls_share_tracker_clamp_ranges(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_target = self.read("overlay/draw_target.cpp")
        keyboard = self.read("keyboard/keyboard_listener.cpp")
        settings_reference = self.read("docs/settings-reference/generate_settings_reference.py")

        for token in (
            "kBodyYOffsetMin",
            "kBodyYOffsetMax",
            "kBodyYOffsetDefault",
            "kHeadYOffsetMin",
            "kHeadYOffsetMax",
            "kHeadYOffsetDefault",
        ):
            self.assertIn(token, config_h)

        self.assertIn("std::clamp", config_cpp)
        self.assertIn("Config::kBodyYOffsetMin", draw_target)
        self.assertIn("Config::kBodyYOffsetMax", draw_target)
        self.assertIn("Config::kHeadYOffsetMin", draw_target)
        self.assertIn("Config::kHeadYOffsetMax", draw_target)
        self.assertIn("Config::kBodyYOffsetMin", keyboard)
        self.assertIn("Config::kHeadYOffsetMin", keyboard)
        self.assertIn("kBodyYOffsetMin = kHeadYOffsetMin", config_h)
        self.assertIn("kBodyYOffsetDefault = 0.15f", config_h)
        self.assertIn("0.05-0.90", settings_reference)
        self.assertIn("0.05-0.55", settings_reference)


if __name__ == "__main__":
    unittest.main()
