import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class NinjaBuildBatContractTests(unittest.TestCase):
    def read_script(self):
        script_path = REPO_ROOT / "build-ninja.bat"
        self.assertTrue(script_path.exists(), f"Missing Ninja build script: {script_path}")
        return script_path.read_text(encoding="utf-8")

    def test_batch_file_generates_repo_local_ninja_file(self):
        script = self.read_script().lower()

        self.assertIn("build\\ninja", script)
        self.assertIn("build.ninja", script)
        self.assertIn("ninja_temp", script)
        self.assertIn("move /y", script)
        self.assertIn("ninja.exe", script)
        self.assertIn("-f", script)

    def test_ninja_targets_cover_dml_cuda_and_worker(self):
        script = self.read_script().lower()

        self.assertIn("build run_dml:", script)
        self.assertIn("build run_cuda:", script)
        self.assertIn("build run_worker:", script)
        self.assertIn("build dml: phony run_dml", script)
        self.assertIn("build cuda: phony run_cuda", script)
        self.assertIn("build worker: phony run_worker", script)
        self.assertIn("build all:", script)
        self.assertIn("default dml", script)

    def test_dml_target_uses_existing_msbuild_project(self):
        script = self.read_script()

        self.assertIn("0BS_box_2.vcxproj", script)
        self.assertIn("/p:Configuration=DML", script)
        self.assertIn("/p:Platform=x64", script)

    def test_cuda_targets_delegate_to_existing_cuda_scripts(self):
        script = self.read_script().replace("/", "\\").lower()

        self.assertIn("cuda\\build-cuda.ps1", script)
        self.assertIn("cuda\\build-yolo-worker.ps1", script)
        self.assertIn("executionpolicy", script)
        self.assertIn("bypass", script)

    def test_script_sets_up_visual_studio_environment_without_rebuilding_dependencies(self):
        script = self.read_script().lower()

        self.assertIn("vsdevcmd.bat", script)
        self.assertIn("msbuild.exe", script)
        self.assertNotIn("opencv.sln", script)
        self.assertNotIn("opencv_world.vcxproj", script)


if __name__ == "__main__":
    unittest.main()
