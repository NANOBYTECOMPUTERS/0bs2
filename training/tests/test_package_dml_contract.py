import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_SCRIPT = REPO_ROOT / "package-dml.ps1"


class PackageDmlContractTest(unittest.TestCase):
    def read_script(self):
        self.assertTrue(PACKAGE_SCRIPT.exists(), f"Missing packaging script: {PACKAGE_SCRIPT}")
        return PACKAGE_SCRIPT.read_text(encoding="utf-8")

    def powershell(self):
        shell = (
            shutil.which("powershell.exe")
            or shutil.which("powershell")
            or shutil.which("pwsh.exe")
            or shutil.which("pwsh")
        )
        self.assertIsNotNone(shell, "PowerShell is required to exercise package-dml.ps1")
        return shell

    def write_file(self, path, content="placeholder"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def run_package(self, source_dir, output_dir, cwd):
        self.assertTrue(PACKAGE_SCRIPT.exists(), f"Missing packaging script: {PACKAGE_SCRIPT}")
        command = [
            self.powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PACKAGE_SCRIPT),
            "-SourceDir",
            str(source_dir),
            "-OutputDir",
            str(output_dir),
        ]
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)

    def run_package_without_args(self, script_path, cwd):
        command = [
            self.powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
        ]
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)

    def test_script_packages_0bs_exe_not_stale_ai_exe(self):
        script = self.read_script()

        self.assertIn("0BS.exe", script)
        self.assertNotIn("ai.exe", script.lower())

    def test_script_defaults_to_dml_build_output(self):
        script = self.read_script().replace("\\", "/").lower()

        self.assertIn("x64/dml", script)

    def test_script_does_not_require_repository_metadata(self):
        script = self.read_script().lower()

        self.assertNotIn(".git", script)
        self.assertNotIn("git ", script)
        self.assertNotIn("github", script)

    def test_script_rejects_source_with_only_stale_ai_exe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "build"
            output_dir = root / "dist" / "0BS"
            self.write_file(source_dir / "ai.exe")
            self.write_file(source_dir / "rzctl.dll")

            result = self.run_package(source_dir, output_dir, root)

            self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("0BS.exe", result.stdout + result.stderr)
            self.assertFalse((output_dir / "ai.exe").exists())
            self.assertFalse((output_dir / "0BS.exe").exists())

    def test_script_can_run_with_default_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "package-dml.ps1"
            script.write_text(self.read_script(), encoding="utf-8")

            self.write_file(root / "x64" / "DML" / "0BS.exe")
            self.write_file(root / "x64" / "DML" / "rzctl.dll")

            result = self.run_package_without_args(script, root)

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertTrue((root / "dist" / "0BS" / "0BS.exe").exists())
            self.assertTrue((root / "dist" / "0BS" / "rzctl.dll").exists())

    def test_script_creates_clean_dist_without_repository_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "plain-build-output"
            output_dir = root / "dist" / "0BS"

            self.write_file(source_dir / "0BS.exe")
            self.write_file(source_dir / "rzctl.dll")
            self.write_file(source_dir / "ghub_mouse.dll")
            self.write_file(source_dir / "opencv_world4130.dll")
            self.write_file(source_dir / "opencv_videoio_ffmpeg4130_64.dll")
            self.write_file(source_dir / "onnxruntime.dll")
            self.write_file(source_dir / "onnxruntime_providers_shared.dll")
            self.write_file(source_dir / "DirectML.dll")
            self.write_file(source_dir / "models" / "example.onnx")
            self.write_file(output_dir / "stale.txt")

            result = self.run_package(source_dir, output_dir, root)

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertTrue((output_dir / "0BS.exe").exists())
            self.assertTrue((output_dir / "rzctl.dll").exists())
            self.assertTrue((output_dir / "ghub_mouse.dll").exists())
            self.assertTrue((output_dir / "opencv_world4130.dll").exists())
            self.assertTrue((output_dir / "opencv_videoio_ffmpeg4130_64.dll").exists())
            self.assertTrue((output_dir / "onnxruntime.dll").exists())
            self.assertTrue((output_dir / "onnxruntime_providers_shared.dll").exists())
            self.assertTrue((output_dir / "DirectML.dll").exists())
            self.assertTrue((output_dir / "config").is_dir())
            self.assertTrue((output_dir / "models" / "example.onnx").exists())
            self.assertFalse((output_dir / "stale.txt").exists())


if __name__ == "__main__":
    unittest.main()
