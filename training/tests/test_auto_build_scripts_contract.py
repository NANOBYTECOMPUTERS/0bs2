import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class AutoBuildScriptsContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_root_launchers_exist_and_delegate_to_powershell(self):
        for relative in (
            "BUILDER.bat",
            "RUN_BUILDER.bat",
            "build_dml.bat",
            "build_cuda.bat",
            "build_no-options.bat",
        ):
            self.assertTrue((REPO_ROOT / relative).exists(), relative)
            text = self.read(relative)
            self.assertIn("powershell", text.lower())
            self.assertIn("-ExecutionPolicy Bypass", text)

    def test_builder_selects_supported_targets_without_opencv_rebuilds(self):
        builder = self.read("BUILDER.ps1")
        self.assertIn('[ValidateSet("DML", "CUDA", "WORKER", "ALL", "")]', builder)
        self.assertIn("Select-BuildTarget", builder)
        self.assertIn("tools\\build_dml.ps1", builder)
        self.assertIn("tools\\build_cuda.ps1", builder)
        self.assertIn("tools\\build_worker.ps1", builder)
        self.assertNotIn("build_opencv", builder.lower())

    def test_build_scripts_use_existing_project_paths(self):
        common = self.read("tools/build_common.ps1")
        dml = self.read("tools/build_dml.ps1")
        cuda = self.read("tools/build_cuda.ps1")
        worker = self.read("tools/build_worker.ps1")
        no_options = self.read("build_no-options.ps1")

        self.assertIn("Resolve-MSBuild", common)
        self.assertIn("Import-VisualStudioEnvironment", common)
        self.assertIn("Invoke-VisualStudioProjectBuild", common)
        self.assertIn("0BS_box_2.vcxproj", dml)
        self.assertIn("/p:Configuration=$Configuration", common)
        self.assertIn("cuda\\build-cuda.ps1", cuda)
        self.assertIn("cuda\\build-yolo-worker.ps1", worker)
        self.assertIn("Runs existing 0BS build paths", no_options)


if __name__ == "__main__":
    unittest.main()
