import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class NeuralTargetingHeadContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_targeting_head_runtime_contract_exists(self):
        header_path = REPO_ROOT / "neural" / "targeting" / "NeuralTargetingHead.h"
        impl_path = REPO_ROOT / "neural" / "targeting" / "NeuralTargetingHead.cpp"
        self.assertTrue(header_path.exists())
        self.assertTrue(impl_path.exists())

        header = header_path.read_text(encoding="utf-8")
        impl = impl_path.read_text(encoding="utf-8")

        for token in (
            "class NeuralTargetingHead",
            "struct Input",
            "struct Output",
            "std::vector<float> predicted_x",
            "std::vector<float> predicted_y",
            "refinement_offset_x",
            "refinement_offset_y",
            "bool load",
            "Output compute",
            "setEnabled",
            "setMaxRefinementPx",
        ):
            self.assertIn(token, header)

        for token in (
            "models/neural_targeting_head.onnx",
            "Ort::Session",
            "NeuralTargetingBaseFeatureCount",
            "refinement_x",
            "refinement_y",
            "max_iterations",
            "outputCount < 3",
            "std::array<int64_t, 2> inputShape",
            "std::array<float, 3>",
        ):
            self.assertIn(token, impl)

    def test_targeting_worker_is_async_cached_and_fail_closed(self):
        header = self.read("neural/targeting/NeuralTargetingHead.h")
        impl = self.read("neural/targeting/NeuralTargetingHead.cpp")

        for token in (
            "class NeuralTargetingWorker",
            "struct Request",
            "struct Result",
            "submit(",
            "tryGet(",
            "clear()",
        ):
            self.assertIn(token, header)

        for token in (
            "std::thread",
            "std::condition_variable",
            "workerLoop",
            "pendingRequests",
            "latestResults",
            "head->compute(request.input)",
            "nextLoadAttempt",
            "return result",
        ):
            self.assertIn(token, impl)

    def test_config_gui_and_runtime_defaults_are_off(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_neural = self.read("overlay/draw_neural.cpp")

        for token in (
            "neural_targeting_enabled",
            "neural_targeting_model_path",
            "neural_targeting_influence",
            "neural_targeting_max_refinement_px",
            "neural_targeting_max_iterations",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_neural)

        self.assertIn("Neural Targeting Head", draw_neural)
        self.assertIn("models/neural_targeting_head.onnx", config_cpp)
        self.assertIn("neural_targeting_enabled = false", config_cpp)

        for token in (
            "neural_targeting_enabled = false",
            'neural_targeting_enabled = get_bool("neural_targeting_enabled", false)',
            "neural_targeting_model_path = \"models/neural_targeting_head.onnx\"",
            "neural_targeting_influence = 0.40f",
            "neural_targeting_max_refinement_px = 35.0f",
        ):
            self.assertIn(token, config_cpp)

    def test_mouse_path_blends_targeting_head_into_feed_forward_only(self):
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")
        pid_cpp = self.read("mouse/PidMouseController.cpp")

        for token in (
            "computeNeuralTargetingInput",
            "consumeNeuralTargetingResult",
            "submitNeuralTargetingRequest",
            "NeuralTargetingWorker::instance()",
            "neural_targeting_enabled",
            "neural_targeting_influence",
            "neural_targeting_max_refinement_px",
            "neuralRefinement",
            "phase2Lead",
            "phase2Lead.first + neuralRefinement.first",
        ):
            self.assertIn(token, mouse_cpp)

        self.assertIn("neuralTargetingPending", mouse_h)
        self.assertIn("neuralTargetingRefinedAimPoint", mouse_h)
        self.assertIn("learnedPredictionLeadX", mouse_cpp)
        self.assertNotIn("observation.x += neural", mouse_cpp)
        self.assertNotIn("command.pixelDx += neural", pid_cpp)

    def test_targeting_safety_gates_are_present(self):
        mouse_cpp = self.read("mouse/mouse.cpp")

        for token in (
            "predictedFutureAgeFrames > 3",
            "maxFirstPointDistance",
            "maxPredictionSpeed",
            "confidenceFloor",
            "directionCosine < -0.35",
            "disagreementCosine",
            "conservativeMaxTotalLead",
            "targetingResult.output.confidence",
            "clampVectorLength",
            "neural_targeting_max_refinement_px",
        ):
            self.assertIn(token, mouse_cpp)

    def test_projects_include_targeting_head_sources(self):
        dml_project = self.read("0BS_box_2.vcxproj")
        cuda_project = self.read("cuda/0BS_cuda.vcxproj")

        for project in (dml_project, cuda_project):
            self.assertIn("NeuralTargetingHead.cpp", project)
            self.assertIn("NeuralTargetingHead.h", project)


if __name__ == "__main__":
    unittest.main()
