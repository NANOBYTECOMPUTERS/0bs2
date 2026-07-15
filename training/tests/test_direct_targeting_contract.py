import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


def join(*parts: str) -> str:
    return "".join(parts)


class DirectTargetingContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_deleted_control_sources_stay_out_of_builds(self):
        removed_files = [
            Path("mouse") / join("Pid", "Mouse", "Controller.cpp"),
            Path("mouse") / join("Pid", "Mouse", "Controller.h"),
            Path("mouse") / join("Smart", "Blender.cpp"),
            Path("mouse") / join("Smart", "Blender.h"),
        ]
        for relative in removed_files:
            self.assertFalse((REPO_ROOT / relative).exists(), str(relative))

        project_text = "\n".join(
            self.read(path)
            for path in (
                "0BS_box_2.vcxproj",
                "0BS_box_2.vcxproj.filters",
                "cuda/0BS_cuda.vcxproj",
                "cuda/0BS_cuda.vcxproj.filters",
            )
        )
        for token in (
            join("Pid", "Mouse", "Controller"),
            join("Smart", "Blender"),
        ):
            self.assertNotIn(token, project_text)

    def test_mouse_thread_dispatches_tracker_aim_points_directly(self):
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")
        loop_cpp = self.read("runtime/mouse_thread_loop.cpp")

        for token in (
            "dispatchTargetMovement(",
            "pixelDeltaToCounts(",
            "movementCountCarryX",
            "movementCountCarryY",
            "queueMove(dx, dy)",
            "recordEgoMotionDelta(pixelDx, pixelDy, now)",
            "blendPredictedAimPoint(",
            "moveMouseTarget(",
            "directMovementTrackId",
        ):
            self.assertIn(token, mouse_cpp + mouse_h)

        self.assertIn("activeTarget->smoothX", loop_cpp)
        self.assertIn("activeTarget->smoothY", loop_cpp)
        self.assertIn("mouseThread.moveMouseTarget(*activeTarget)", loop_cpp)
        self.assertIn("return { pixelDx, pixelDy };", mouse_cpp)

        for token in (
            join("Pid", "Mouse", "Controller"),
            join("Smart", "Blender"),
            join("feed", "_forward"),
            join("smart", "_blending"),
            "fov_x",
            "fov_y",
            "fovX",
            "fovY",
            "degToCounts(",
            "GameProfile",
            "currentProfile(",
            "game_profiles",
            "active_game",
        ):
            self.assertNotIn(token, mouse_cpp + mouse_h + loop_cpp)

    def test_direct_targeting_settings_are_wired_through_config_and_ui(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        generator = self.read("docs/settings-reference/generate_settings_reference.py")

        for key in (
            "target_deadzone_px",
            "target_max_pixel_step",
            "target_output_scale",
            "target_calibrated_pixel_counts_enabled",
            "target_counts_per_pixel_x",
            "target_counts_per_pixel_y",
            "target_prediction_blend",
            "target_prediction_max_lead_px",
        ):
            self.assertIn(key, config_h)
            self.assertIn(key, config_cpp)
            self.assertIn(key, draw_mouse)
            self.assertIn(key, generator)

        self.assertIn('OverlayUI::BeginSection("Direct Targeting Movement"', draw_mouse)
        self.assertIn('ImGui::Checkbox("Calibrated pixel counts"', draw_mouse)
        self.assertIn('MERGE_FIELD("target_calibrated_pixel_counts_enabled"', config_cpp)
        self.assertIn('"target_counts_per_pixel_x = "', config_cpp)
        self.assertIn("config.target_calibrated_pixel_counts_enabled", self.read("mouse/mouse.cpp"))


if __name__ == "__main__":
    unittest.main()
