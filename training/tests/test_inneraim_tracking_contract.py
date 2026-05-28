import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "BoxTarget.h").exists()
)


class InnerAimTrackingContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_inneraim_track_state_is_resolution_aware(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "struct InnerAimTrack",
            "smoothX",
            "smoothY",
            "radius",
            "consistencyScore",
            "aim::AimKalman2D kalman",
            "scaleFactor() const",
            "computeAssociationCost(",
        ):
            self.assertIn(token, target_h)

        for token in (
            "config.detection_resolution / 320.0f",
            "18.0f * scaleFactor()",
            "45.0f * scaleFactor()",
            "0.4f * scaleFactor()",
            "16.0f * scaleFactor()",
            "newConf > track.confidence + 0.38f",
            "newConf > 0.72f",
        ):
            self.assertIn(token, target_cpp)

    def test_inneraim_uses_class_offsets_and_head_priority(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "innerAimX",
            "innerAimY",
            "innerAimClassId",
        ):
            self.assertIn(token, target_h)

        for token in (
            "classId == config.class_head",
            "classId == config.class_player",
            "config.head_y_offset",
            "config.body_y_offset",
            "d.innerAimX = playerHeadPivotX[i]",
            "d.innerAimY = playerHeadPivotY[i]",
            "d.innerAimClassId = config.class_head",
            "d.innerAimX,",
            "d.innerAimY",
        ):
            self.assertIn(token, target_cpp)

    def test_association_kalman_and_pid_handoff_use_inneraim(self):
        target_cpp = self.read("mouse/BoxTarget.cpp")
        target_h = self.read("mouse/BoxTarget.h")
        mouse_loop = self.read("runtime/mouse_thread_loop.cpp")

        for token in (
            "0.22 * (1.0f - iouScore)",
            "0.48 * normDist",
            "0.30 * (1.0 - consistencyBonus)",
            "std::min(1.0, dist / std::max(12.0",
            "track.consistencyScore * 0.8",
            "velocity * 0.011",
            "process_noise_position",
        ):
            self.assertIn(token, target_cpp)

        for token in ("innerAimX", "innerAimY", "innerAimRadius", "consistencyScore"):
            self.assertIn(token, target_h)

        self.assertIn(
            "void updateInnerAim(InnerAimTrack& track, const cv::Rect& det, float conf, double dt)",
            target_h,
        )
        self.assertIn("shouldAcceptAsNewLock(", target_h)
        self.assertIn("activeTarget->smoothX", mouse_loop)
        self.assertIn("activeTarget->smoothY", mouse_loop)
        self.assertIn("moveMousePivot(", mouse_loop)
        self.assertIn("0.0, 0.0", mouse_loop)

    def test_inneraim_kalman_respects_mouse_tab_checkbox(self):
        target_cpp = self.read("mouse/BoxTarget.cpp")

        self.assertIn("settings.enabled = config.kalman_enabled", target_cpp)
        self.assertIn("if (!config.kalman_enabled)", target_cpp)
        self.assertIn("track.kalman.reset()", target_cpp)
        self.assertIn("velocityLeadSeconds", target_cpp)

    def test_overlay_draws_inneraim_cross_and_radius(self):
        overlay = self.read("runtime/game_overlay_loop.cpp")

        for token in (
            "innerAimRadius",
            "innerAimX",
            "innerAimY",
            "AddLine",
            "AddCircle",
            "consistencyScore",
        ):
            self.assertIn(token, overlay)


if __name__ == "__main__":
    unittest.main()
