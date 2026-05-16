import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class NeuralTrackerContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_neural_tracker_module_exists_and_is_optional(self):
        header = self.read("neural/NeuralTracker.h")
        impl = self.read("neural/NeuralTracker.cpp")

        self.assertIn("INeuralTracker", header)
        self.assertIn("NeuralTrackerFeatureCount", header)
        self.assertIn("createOnnxNeuralTracker", header)
        self.assertIn("resolveNeuralModelPath", impl)
        self.assertIn("ONNX model missing", impl)
        self.assertIn("return nullptr", impl)

    def test_config_exposes_neural_tracker_and_debug_flags(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        for name in (
            "neural_tracker_enabled",
            "neural_tracker_model_path",
            "neural_tracker_blend",
            "neural_tracker_log_enabled",
            "neural_tracker_debug_enabled",
        ):
            self.assertIn(name, config_h)
            self.assertIn(name, config_cpp)

    def test_tracker_uses_neural_score_as_association_helper(self):
        target_cpp = self.read("mouse/BoxTarget.cpp")

        self.assertIn("neural/NeuralTracker.h", target_cpp)
        self.assertIn("buildNeuralFeatures", target_cpp)
        self.assertIn("neuralScore", target_cpp)
        self.assertIn("neuralBonus", target_cpp)

    def test_gui_has_neural_tab_and_debug_toggles(self):
        draw_settings = self.read("overlay/draw_settings.h")
        overlay = self.read("overlay/overlay.cpp")
        neural_gui = self.read("overlay/draw_neural.cpp")
        debug_gui = self.read("overlay/draw_debug.cpp")

        self.assertIn("draw_neural", draw_settings)
        self.assertIn('"Neural"', overlay)
        self.assertIn("Neural Tracker", neural_gui)
        self.assertIn("neural_tracker_enabled", neural_gui)
        self.assertIn("neural_tracker_log_enabled", debug_gui)
        self.assertIn("neural_tracker_debug_enabled", debug_gui)

    def test_build_files_include_neural_sources(self):
        project_texts = [
            self.read("0BS_box_2.vcxproj"),
            self.read("0BS_box_2.vcxproj.filters"),
        ]
        cmake_path = REPO_ROOT / "CMakeLists.txt"
        if cmake_path.exists():
            project_texts.append(cmake_path.read_text(encoding="utf-8"))

        for project_text in project_texts:
            self.assertIn("neural", project_text)
            self.assertIn("NeuralTracker.cpp", project_text)
            self.assertIn("NeuralTracker.h", project_text)


if __name__ == "__main__":
    unittest.main()
