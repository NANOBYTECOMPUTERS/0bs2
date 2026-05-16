import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class PidFeedForwardContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_pid_settings_expose_gated_feed_forward_controls(self):
        header = self.read("mouse/PidMouseController.h")
        controller = self.read("mouse/PidMouseController.cpp")

        self.assertIn("feedForwardEnabled", header)
        self.assertIn("feedForwardGain", header)
        self.assertIn("feedForwardLookaheadMs", header)
        self.assertIn("feedForwardMaxStep", header)
        self.assertIn("feedForwardMinSpeed", header)
        self.assertIn("feedForwardConfidenceFloor", header)

        self.assertIn("computeFeedForward", header)
        self.assertIn("targetVx", controller)
        self.assertIn("targetAx", controller)
        self.assertIn("feedForwardScale", controller)
        self.assertIn("closingRate", controller)
        self.assertIn("overshot", controller)

    def test_target_motion_estimate_compensates_for_own_camera_movement(self):
        header = self.read("mouse/PidMouseController.h")
        controller = self.read("mouse/PidMouseController.cpp")

        self.assertIn("movementSinceObservationX", header)
        self.assertIn("movementSinceObservationY", header)
        self.assertIn("observation.x - latest.x + movementSinceObservationX", controller)
        self.assertIn("observation.y - latest.y + movementSinceObservationY", controller)
        self.assertIn("movementSinceObservationX += outX", controller)
        self.assertIn("movementSinceObservationY += outY", controller)

    def test_feed_forward_brakes_after_overshoot_and_rejects_bad_confidence(self):
        header = self.read("mouse/PidMouseController.h")
        controller = self.read("mouse/PidMouseController.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")

        self.assertIn("feedForwardCooldownSeconds", header)
        self.assertIn("FeedForwardOvershootCooldownSeconds", controller)
        self.assertIn("feedForwardCooldownSeconds = std::max", controller)
        self.assertIn("feedForwardCooldownSeconds > 0.0", controller)
        self.assertIn("std::isfinite(latest.confidence)", controller)
        self.assertIn("std::isfinite(confidence)", mouse_cpp)

    def test_config_and_ui_wire_feed_forward_controls(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")
        overlay_cpp = self.read("overlay/draw_mouse.cpp")

        for source in (config_h, config_cpp, mouse_cpp, overlay_cpp):
            self.assertIn("pid_feed_forward_enabled", source)
            self.assertIn("pid_feed_forward_gain", source)
            self.assertIn("pid_feed_forward_lookahead_ms", source)
            self.assertIn("pid_feed_forward_max_step", source)
            self.assertIn("pid_feed_forward_min_speed", source)
            self.assertIn("pid_feed_forward_confidence_floor", source)

        self.assertIn('ImGui::Checkbox("Feed-forward prediction"', overlay_cpp)
        self.assertIn('ImGui::SliderFloat("Feed-forward gain"', overlay_cpp)
        self.assertIn("feedForwardEnabled = config.pid_feed_forward_enabled", mouse_cpp)


if __name__ == "__main__":
    unittest.main()
