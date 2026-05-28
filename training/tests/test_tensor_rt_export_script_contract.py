import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class TensorRtExportScriptContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_export_script_exists_and_uses_trtexec(self):
        script_path = REPO_ROOT / "cuda" / "export-engine.ps1"
        self.assertTrue(script_path.exists())

        script = self.read("cuda/export-engine.ps1")
        self.assertIn("trtexec.exe", script)
        self.assertIn("TensorRTDir", script)
        self.assertIn("TensorRT-*", script)
        self.assertIn("DryRun", script)
        self.assertIn("No external dependencies", script)

    def test_export_script_defaults_to_sibling_engine(self):
        script = self.read("cuda/export-engine.ps1")

        self.assertIn("replace_extension", script)
        self.assertIn(".engine", script)
        self.assertIn("Resolve-Path", script)
        self.assertIn("Test-Path", script)
        self.assertIn("--onnx=", script)
        self.assertIn("--saveEngine=", script)

    def test_export_script_exposes_precision_and_dynamic_shape_options(self):
        script = self.read("cuda/export-engine.ps1")

        self.assertIn("UseFp16", script)
        self.assertIn("UseFp8", script)
        self.assertIn("MinShapes", script)
        self.assertIn("OptShapes", script)
        self.assertIn("MaxShapes", script)
        self.assertIn("--fp16", script)
        self.assertIn("--fp8", script)
        self.assertIn("--minShapes=", script)
        self.assertIn("--optShapes=", script)
        self.assertIn("--maxShapes=", script)

    def test_cuda_readme_documents_export_script(self):
        readme = self.read("cuda/README.md")

        self.assertIn("export-engine.ps1", readme)
        self.assertIn("training/models/neural_tracker.onnx", readme)
        self.assertIn("-DryRun", readme)


if __name__ == "__main__":
    unittest.main()
