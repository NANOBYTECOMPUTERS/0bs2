import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class CMakeModernizationContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_cmake_path_is_additive_and_test_focused(self):
        cmake = self.read("CMakeLists.txt")
        presets = self.read("CMakePresets.json")
        docs = self.read("docs/build-modernization.md")

        self.assertIn("project(OBS2", cmake)
        self.assertIn("obs2_targeting_core", cmake)
        self.assertIn("obs2_targeting_tests", cmake)
        self.assertIn("obs2_targeting_convergence_tests", cmake)
        self.assertIn("obs2_capture_geometry_tests", cmake)
        self.assertIn("obs2_postprocess_tests", cmake)
        self.assertIn("obs2_tracker_state_tests", cmake)
        self.assertIn("OBS2_ENABLE_CLANG_TIDY", cmake)
        self.assertIn("OBS2_USE_CUDA", cmake)
        self.assertIn("OBS2_USE_DIRECTML", cmake)
        self.assertIn("vs2026-tests", presets)
        self.assertIn("vs2026-cuda-tests", presets)
        self.assertIn('"OBS2_USE_CUDA": "ON"', presets)
        self.assertIn("Existing MSBuild projects remain authoritative", presets)
        self.assertIn("existing MSBuild projects and scripts remain the authoritative application build", docs)

    def test_native_tests_cover_targeting_math_without_external_sdks(self):
        native_tests = self.read("tests/cpp/targeting_math_tests.cpp")
        cmake = self.read("CMakeLists.txt")

        for token in (
            "aim_kalman.h",
            "aim_imm.h",
            "ego_motion_compensator.h",
            "testKalmanSettingsClamp",
            "testKalmanRuntimeSweepPredictionRamp",
            "testImmProducesFiniteProbabilitiesAndPrediction",
            "testEgoMotionConsumesTimeWindowAndClamps",
        ):
            self.assertIn(token, native_tests)

        self.assertNotIn("opencv2/", native_tests)
        self.assertNotIn("NvInfer", native_tests)
        self.assertIn("target_include_directories(obs2_targeting_core", cmake)

    def test_native_tests_cover_targeting_convergence_tuning(self):
        cmake = self.read("CMakeLists.txt")
        convergence_tests = self.read("tests/cpp/targeting_convergence_tests.cpp")
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        self.assertIn("obs2_targeting_convergence_tests", cmake)
        for token in (
            "simulateStreamStep",
            "testStreamSharpnessConvergesWithoutSpeedCapDependency",
            "testSeededLatencySweepImprovesPredictionConvergence",
            "testTunedKalmanDoesNotWorsenReversalRecovery",
            "tunedConvergenceSettings",
        ):
            self.assertIn(token, convergence_tests)

        for token in (
            "kRuntimeLatencySweepEnabledDefault = true",
            "kKalmanProcessNoiseVelocityDefault = 3200.0f",
            "kKalmanMeasurementNoiseDefault = 18.0f",
            "kKalmanVelocityDampingDefault = 0.04f",
            "kKalmanAcquisitionFramesDefault = 3",
            "kTargetStreamSharpnessDefault = 24.0f",
            "kTargetMinStreamConfidenceDefault = 0.55f",
        ):
            self.assertIn(token, config_h)

        for token in (
            "runtime_latency_sweep_enabled = kRuntimeLatencySweepEnabledDefault",
            "kalman_acquisition_frames = kKalmanAcquisitionFramesDefault",
            'get_double("target_stream_sharpness", kTargetStreamSharpnessDefault)',
            'get_double("target_min_stream_confidence", kTargetMinStreamConfidenceDefault)',
        ):
            self.assertIn(token, config_cpp)

    def test_extended_native_tests_cover_pipeline_state_with_opencv(self):
        cmake = self.read("CMakeLists.txt")
        deps = self.read("cmake/OBS2Dependencies.cmake")
        capture_tests = self.read("tests/cpp/capture_geometry_tests.cpp")
        postprocess_tests = self.read("tests/cpp/postprocess_tests.cpp")
        tracker_tests = self.read("tests/cpp/tracker_state_tests.cpp")
        docs = self.read("docs/build-modernization.md")

        for token in (
            "obs2_configure_opencv_dependency",
            "OBS2::OpenCV",
            "obs2_copy_opencv_runtime",
            "obs2_apply_opencv_test_environment",
            "CUDNN",
        ):
            self.assertIn(token, deps + cmake)

        for token in (
            "CaptureFrameGeometry::FromCenterCrop",
            "modelToScreenPoint",
            "screenToModelPoint",
            "isInsideScreenCrop",
        ):
            self.assertIn(token, capture_tests)

        for token in (
            "postProcessYoloScaled",
            "postProcessYoloDML",
            "NMS(",
            "YOLO_ANNOTATION_WORKER",
            "USE_CUDA",
        ):
            self.assertIn(token, postprocess_tests + cmake)

        for token in (
            "MultiTargetTracker",
            "getLockedTarget",
            "getDebugTracks",
            "testTrackerLocksAndMaintainsIdThroughMatchedMovement",
            "testTrackerKeepsLockedTargetDuringShortOcclusion",
            "testTrackerGatesLowConfidenceAndPromotesTentativeTrack",
        ):
            self.assertIn(token, tracker_tests)

        self.assertIn("postprocess decoding", docs)
        self.assertIn("tracker lock/lost/confirmed transitions", docs)
        self.assertIn("vs2026-cuda-tests", docs)

    def test_existing_build_entrypoints_remain_in_place(self):
        build_ninja = self.read("build-ninja.bat")
        self.assertTrue((REPO_ROOT / "0BS_box_2.vcxproj").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "0BS_cuda.vcxproj").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "yolo_annotation_worker.vcxproj").exists())
        self.assertIn("build run_dml:", build_ninja)
        self.assertIn("build run_cuda:", build_ninja)
        self.assertIn("build run_worker:", build_ninja)


if __name__ == "__main__":
    unittest.main()
