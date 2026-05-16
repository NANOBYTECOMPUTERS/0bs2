import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class BuildPackagesContractTest(unittest.TestCase):
    def test_vcxproj_uses_root_packages_folder(self):
        project = (REPO_ROOT / "0BS_box_2.vcxproj").read_text(encoding="utf-8")

        self.assertNotIn("..\\packages\\", project)
        self.assertNotIn("x64\\DML\\packages\\", project)
        self.assertIn("packages\\Microsoft.Windows.CppWinRT.2.0.240405.15\\build\\native\\Microsoft.Windows.CppWinRT.props", project)
        self.assertIn("packages\\Microsoft.Windows.CppWinRT.2.0.240405.15\\build\\native\\Microsoft.Windows.CppWinRT.targets", project)
        self.assertIn("packages\\Microsoft.AI.DirectML.1.15.4\\build\\Microsoft.AI.DirectML.props", project)
        self.assertIn("packages\\Microsoft.AI.DirectML.1.15.4\\build\\Microsoft.AI.DirectML.targets", project)
        self.assertIn("packages\\Microsoft.ML.OnnxRuntime.DirectML.1.22.0\\build\\native\\Microsoft.ML.OnnxRuntime.DirectML.props", project)
        self.assertIn("packages\\Microsoft.ML.OnnxRuntime.DirectML.1.22.0\\build\\native\\Microsoft.ML.OnnxRuntime.DirectML.targets", project)

    def test_project_package_imports_exist_on_disk(self):
        project = (REPO_ROOT / "0BS_box_2.vcxproj").read_text(encoding="utf-8")
        imports = re.findall(r'<Import Project="([^"]*packages\\[^"]*)"', project)

        self.assertGreaterEqual(len(imports), 6)
        for import_path in imports:
            self.assertTrue((REPO_ROOT / import_path).exists(), import_path)

    def test_msvc_experimental_coroutine_warning_is_silenced(self):
        project = (REPO_ROOT / "0BS_box_2.vcxproj").read_text(encoding="utf-8")

        self.assertIn("_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS", project)

    def test_dml_build_outputs_0bs_executable(self):
        project = (REPO_ROOT / "0BS_box_2.vcxproj").read_text(encoding="utf-8")
        mouse_overlay = (REPO_ROOT / "overlay" / "draw_mouse.cpp").read_text(encoding="utf-8")

        self.assertIn("<TargetName>0BS</TargetName>", project)
        self.assertNotIn("<TargetName>ai</TargetName>", project)
        self.assertNotIn("ai.exe", mouse_overlay)
        self.assertIn("0BS.exe", mouse_overlay)


if __name__ == "__main__":
    unittest.main()
