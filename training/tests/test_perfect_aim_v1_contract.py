import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class PerfectAimV1ContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_config_and_neural_gui_expose_v1_presets_and_telemetry(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_neural = self.read("overlay/draw_neural.cpp")

        for token in (
            "neural_control_preset",
            "neural_control_telemetry_overlay_enabled",
            "neural_control_telemetry_logging_enabled",
            "neural_control_telemetry_log_path",
            "neural_control_telemetry_log_interval_ms",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_neural)

        self.assertIn('neural_control_preset = "Balanced"', config_cpp)
        self.assertIn("neural_control_telemetry_overlay_enabled = false", config_cpp)
        self.assertIn("neural_control_telemetry_logging_enabled = false", config_cpp)
        self.assertIn("Perfect Aim v1.0 Presets", draw_neural)
        for preset in ("Balanced", "Aggressive", "Smooth", "Sniper"):
            self.assertIn(preset, draw_neural)
        self.assertIn("Base prediction influence", draw_neural)
        self.assertIn("Neural telemetry overlay", draw_neural)
        self.assertIn("Neural telemetry logging", draw_neural)

    def test_adaptive_influence_has_hysteresis_and_track_reset(self):
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for token in (
            "adaptivePredictionInfluenceActive",
            "AdaptiveInfluenceActivateThreshold",
            "AdaptiveInfluenceDeactivateThreshold",
            "resetAdaptivePredictionInfluence",
        ):
            self.assertIn(token, mouse_h + mouse_cpp)

        self.assertIn("rawInfluence >= AdaptiveInfluenceActivateThreshold", mouse_cpp)
        self.assertIn("rawInfluence <= AdaptiveInfluenceDeactivateThreshold", mouse_cpp)
        self.assertIn("adaptivePredictionInfluenceActive = false", mouse_cpp)
        self.assertIn("trackId != adaptivePredictionTrackId", mouse_cpp)

    def test_neural_refinement_rejects_opposing_pid_direction(self):
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")

        self.assertIn("rejectNeuralRefinementAgainstPidDirection", mouse_h)
        self.assertIn("rejectNeuralRefinementAgainstPidDirection", mouse_cpp)
        self.assertIn("NeuralRefinementOpposingPidCosineThreshold", mouse_cpp)
        self.assertIn("NeuralRefinementLowConfidenceThreshold", mouse_cpp)
        self.assertIn("NeuralRefinementDisagreementScale", mouse_cpp)
        self.assertIn("pidDirectionCosine < NeuralRefinementOpposingPidCosineThreshold", mouse_cpp)
        self.assertIn("targetingResult.output.confidence < NeuralRefinementLowConfidenceThreshold", mouse_cpp)
        self.assertIn("neuralRefinement = rejectNeuralRefinementAgainstPidDirection", mouse_cpp)

    def test_smart_blender_tracks_jitter_and_penalizes_oscillation(self):
        blender_h = self.read("mouse/SmartBlender.h")
        blender_cpp = self.read("mouse/SmartBlender.cpp")
        pid_h = self.read("mouse/PidMouseController.h")
        pid_cpp = self.read("mouse/PidMouseController.cpp")

        for token in (
            "feedForwardX",
            "feedForwardY",
            "learnedFeedForwardX",
            "learnedFeedForwardY",
            "neuralRefinementX",
            "neuralRefinementY",
            "jitterScore",
            "oscillationPenalty",
        ):
            self.assertIn(token, blender_h)

        self.assertIn("updateJitterScore", blender_cpp)
        self.assertIn("jitterScore +=", blender_cpp)
        self.assertIn("jitterScore *= 0.90", blender_cpp)
        self.assertIn("const double oscillationPenalty", blender_cpp)
        self.assertIn("output.jitterScore = jitterScore", blender_cpp)
        self.assertIn("output.oscillationPenalty = oscillationPenalty", blender_cpp)
        self.assertIn("smartBlendJitterScore", pid_h)
        self.assertIn("blendInput.learnedFeedForwardX = command.learnedFeedForwardX", pid_cpp)
        self.assertIn("command.smartBlendJitterScore = blendOutput.jitterScore", pid_cpp)

    def test_mouse_publishes_neural_control_telemetry_for_overlay_and_logging(self):
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")
        overlay_loop = self.read("runtime/game_overlay_loop.cpp")

        for token in (
            "struct NeuralControlTelemetry",
            "getNeuralControlTelemetry",
            "publishNeuralControlTelemetry",
            "adaptiveInfluence",
            "predictionLeadX",
            "neuralRefinementX",
            "totalLeadX",
            "jitterScore",
        ):
            self.assertIn(token, mouse_h + mouse_cpp)

        self.assertIn("neural_control_telemetry_logging_enabled", mouse_cpp)
        self.assertIn("neural_control_telemetry_log_interval_ms", mouse_cpp)
        self.assertIn("logs/neural_control_telemetry.csv", mouse_cpp)
        self.assertIn("neural_control_telemetry_overlay_enabled", overlay_loop)
        self.assertIn("getNeuralControlTelemetry", overlay_loop)
        self.assertIn("Adaptive influence", overlay_loop)
        self.assertIn("Predicted lead", overlay_loop)

    def test_readme_documents_perfect_aim_v1_default_off_boundary(self):
        readme = self.read("README.md")
        self.assertIn("Perfect Aim v1.0", readme)
        self.assertIn("advisory only", readme)
        self.assertIn("default OFF", readme)
        self.assertIn("PID/Kalman remains the convergence owner", readme)


if __name__ == "__main__":
    unittest.main()
