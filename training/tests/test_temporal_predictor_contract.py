import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class TemporalPredictorContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_temporal_predictor_runtime_contract_exists(self):
        header = self.read("neural/TemporalPredictor.h")
        impl = self.read("neural/TemporalPredictor.cpp")

        for token in (
            "class TemporalPredictor",
            "struct Input",
            "struct Output",
            "std::vector<float> history",
            "history_length = 12",
            "prediction_horizon = 16",
            "bool loadModel",
            "Output predict",
            "setHistoryLength",
            "setPredictionHorizon",
            "available() const",
        ):
            self.assertIn(token, header)

        for token in (
            "Ort::Session",
            "models/temporal_predictor.onnx",
            "std::chrono::steady_clock",
            "future_x",
            "future_y",
            "valid = false",
        ):
            self.assertIn(token, impl)

    def test_tracker_builds_history_and_caches_predictions_per_track(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "TrackHistorySample",
            "history",
            "temporalPrediction",
            "lastTemporalPredictionFrame",
            "temporalPredictionValid",
            "std::vector<std::pair<double, double>> predictedFuture",
        ):
            self.assertIn(token, target_h)

        for token in (
            "neural/TemporalPredictor.h",
            "buildTemporalPredictorInput",
            "updateTemporalPrediction",
            "config.temporal_prediction_enabled",
            "config.temporal_prediction_interval_frames",
            "TemporalPredictionWorker::instance()",
            "worker.submit",
            "worker.tryGet",
            "workerResult.output.future_x",
            "workerResult.output.future_y",
        ):
            self.assertIn(token, target_cpp)

        self.assertIn("createTemporalPredictor", self.read("neural/TemporalPredictor.cpp"))

    def test_mouse_loop_uses_learned_prediction_when_available(self):
        loop = self.read("runtime/mouse_thread_loop.cpp")

        self.assertIn("lockInfo.predictedFuture", loop)
        self.assertIn("storeFuturePositions(lockInfo.predictedFuture)", loop)
        self.assertIn("predictFuturePositions", loop)

    def test_overlay_draws_temporal_predictions_separately(self):
        overlay = self.read("runtime/game_overlay_loop.cpp")

        self.assertIn("temporalFuture", overlay)
        self.assertIn("temporalPredictionValid", overlay)
        self.assertIn("ARGB(220, 120, 255, 210)", overlay)
        self.assertIn("modelToScreenPoint", overlay)

    def test_config_and_gui_expose_temporal_prediction_controls(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_neural = self.read("overlay/draw_neural.cpp")

        for token in (
            "temporal_prediction_enabled",
            "temporal_prediction_model_path",
            "temporal_prediction_history_length",
            "temporal_prediction_horizon",
            "temporal_prediction_interval_frames",
            "temporal_prediction_feed_forward_enabled",
            "temporal_prediction_influence",
            "adaptive_prediction_enabled",
            "base_prediction_influence",
            "temporal_prediction_max_lead_px",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_neural)

        self.assertIn("Learned Temporal Predictor", draw_neural)
        self.assertIn("Prediction feed-forward", draw_neural)
        self.assertIn("Base prediction influence", draw_neural)
        self.assertIn("Adaptive prediction", draw_neural)
        self.assertIn("Max prediction lead", draw_neural)
        self.assertIn("models/temporal_predictor.onnx", config_cpp)

    def test_predictor_is_default_off_rate_limited_and_fail_closed(self):
        config_cpp = self.read("config/config.cpp")
        target_cpp = self.read("mouse/BoxTarget.cpp")
        predictor_cpp = self.read("neural/TemporalPredictor.cpp")

        self.assertIn("temporal_prediction_enabled = false", config_cpp)
        self.assertIn("temporal_prediction_interval_frames = 2", config_cpp)
        self.assertIn("updateFrameCounter_ - t.lastTemporalPredictionFrame < interval", target_cpp)
        self.assertIn("worker.clear()", target_cpp)
        self.assertIn("nextLoadAttempt", predictor_cpp)
        self.assertIn("!predictor || !predictor->available()", predictor_cpp)
        self.assertIn("t.temporalPredictionValid = false", target_cpp)
        self.assertIn("inference_ms", predictor_cpp)
        self.assertIn("SetIntraOpNumThreads(1)", predictor_cpp)

    def test_temporal_predictor_inference_is_async_cached(self):
        header = self.read("neural/TemporalPredictor.h")
        impl = self.read("neural/TemporalPredictor.cpp")
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "class TemporalPredictionWorker",
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
            "predictor->predict(request.input)",
        ):
            self.assertIn(token, impl)

        self.assertIn("lastTemporalPredictionRequestFrame", target_h)
        self.assertIn("temporalPredictionPending", target_h)
        self.assertIn("TemporalPredictionWorker::instance()", target_cpp)
        self.assertIn("worker.submit", target_cpp)
        self.assertIn("worker.tryGet", target_cpp)
        self.assertNotIn("auto prediction = predictor->predict(input)", target_cpp)

    def test_projects_include_temporal_predictor_sources(self):
        dml_project = self.read("0BS_box_2.vcxproj")
        cuda_project = self.read("cuda/0BS_cuda.vcxproj")

        for project in (dml_project, cuda_project):
            self.assertIn("TemporalPredictor.cpp", project)
            self.assertIn("TemporalPredictor.h", project)

    def test_source_defaults_keep_temporal_prediction_opt_in(self):
        config = self.read("config/config.cpp")
        for token in (
            "temporal_prediction_enabled = false",
            'temporal_prediction_enabled = get_bool("temporal_prediction_enabled", false)',
            "temporal_prediction_history_length = 12",
            "temporal_prediction_horizon = 16",
            "temporal_prediction_interval_frames = 2",
            "temporal_prediction_feed_forward_enabled = false",
            "base_prediction_influence = 0.30f",
            "temporal_prediction_influence = base_prediction_influence",
            "temporal_prediction_max_lead_px = 45.0f",
        ):
            self.assertIn(token, config)

    def test_prediction_feed_forward_is_safety_gated_before_pid(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")
        loop = self.read("runtime/mouse_thread_loop.cpp")

        for token in (
            "predictedFutureAgeFrames",
            "targetVelocityX",
            "targetVelocityY",
        ):
            self.assertIn(token, target_h)
            self.assertIn(token, target_cpp)

        for token in (
            "computePredictionFeedForwardLead",
            "temporal_prediction_feed_forward_enabled",
            "base_prediction_influence",
            "temporal_prediction_max_lead_px",
            "predictedFutureAgeFrames > 3",
            "maxFirstPointDistance",
            "maxPredictionSpeed",
            "confidenceFloor",
            "directionCosine < -0.35",
            "clampVectorLength",
            "smoothStep01",
        ):
            self.assertIn(token, mouse_cpp)

        self.assertIn("computePredictionFeedForwardLead", mouse_h)
        self.assertIn("learnedPredictionLead", loop)
        self.assertIn("activeTarget->smoothX", loop)
        self.assertIn("activeTarget->smoothY", loop)

    def test_adaptive_prediction_influence_has_primary_config_and_legacy_aliases(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_neural = self.read("overlay/draw_neural.cpp")
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for token in (
            "adaptive_prediction_enabled",
            "base_prediction_influence",
            "temporal_prediction_adaptive_influence_enabled",
            "temporal_prediction_adaptive_ema_alpha",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_neural)

        self.assertIn("adaptive_prediction_enabled = true", config_cpp)
        self.assertIn("base_prediction_influence = 0.30f", config_cpp)
        self.assertIn('get_bool("adaptive_prediction_enabled"', config_cpp)
        self.assertIn('get_double("base_prediction_influence"', config_cpp)
        self.assertIn("temporal_prediction_adaptive_influence_enabled = adaptive_prediction_enabled", config_cpp)
        self.assertIn("temporal_prediction_influence = base_prediction_influence", config_cpp)
        self.assertIn("temporal_prediction_adaptive_ema_alpha = 0.62f", config_cpp)
        self.assertIn("Adaptive prediction", draw_neural)
        self.assertIn("Base prediction influence", draw_neural)
        self.assertIn("Adaptive influence EMA", draw_neural)

        for token in (
            "adaptivePredictionInfluenceEma",
            "adaptivePredictionTrackId",
            "resetAdaptivePredictionInfluence",
            "computeAdaptivePredictionInfluence",
        ):
            self.assertIn(token, mouse_h)
            self.assertIn(token, mouse_cpp)

        for token in (
            "smoothStepRange(30.0, 120.0",
            "relativeSpeed / 1100.0",
            "egoMotionCameraVelocityPxPerSec",
            "relativeSpeed",
            "egoStability",
            "confidenceWeight",
            "std::clamp(rawInfluence, 0.0, 1.0)",
            "adaptivePredictionInfluenceEma += (rawInfluence - adaptivePredictionInfluenceEma) * emaAlpha",
            "config.adaptive_prediction_enabled",
            "return baseInfluence",
        ):
            self.assertIn(token, mouse_cpp)

    def test_pid_uses_learned_prediction_only_inside_feed_forward(self):
        pid_h = self.read("mouse/PidMouseController.h")
        pid_cpp = self.read("mouse/PidMouseController.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for token in (
            "learnedPredictionLeadX",
            "learnedPredictionLeadY",
            "learnedPredictionLeadValid",
        ):
            self.assertIn(token, pid_h)
            self.assertIn(token, mouse_cpp)

        for token in (
            "latest.learnedPredictionLeadValid",
            "learnedStepX",
            "learnedStepY",
            "command.learnedFeedForwardX",
            "command.learnedFeedForwardY",
            "stepX += learnedStepX",
            "stepY += learnedStepY",
        ):
            self.assertIn(token, pid_cpp)

        self.assertNotIn("observation.x += learned", mouse_cpp)
        self.assertNotIn("command.pixelDx += learned", pid_cpp)


if __name__ == "__main__":
    unittest.main()
