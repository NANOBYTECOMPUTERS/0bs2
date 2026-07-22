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
        self.assertIn("OBS2_ENABLE_CLANG_TIDY", cmake)
        self.assertIn("OBS2_USE_CUDA", cmake)
        self.assertIn("OBS2_USE_DIRECTML", cmake)
        self.assertIn("vs2026-tests", presets)
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
