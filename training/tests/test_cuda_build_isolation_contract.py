import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class CudaBuildIsolationContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_dml_project_remains_dml_only(self):
        project = self.read("0BS_box_2.vcxproj")
        self.assertIn('<ProjectConfiguration Include="DML|x64">', project)
        self.assertNotIn('<ProjectConfiguration Include="CUDA|x64">', project)
        self.assertNotIn("USE_CUDA", project)
        self.assertNotIn("CudaCompile", project)
        self.assertNotIn("trt_detector.cpp", project)
        self.assertNotIn("cuda_preprocess.cu", project)
        self.assertNotIn("depth_anything_trt.cpp", project)
        self.assertNotIn("gpu_resource_manager.cpp", project)

    def test_cuda_project_lives_in_own_folder(self):
        self.assertTrue((REPO_ROOT / "cuda" / "0BS_cuda.vcxproj").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "0BS_cuda.vcxproj.filters").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "0BS_cuda.sln").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "build-cuda.ps1").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "README.md").exists())

    def test_cuda_project_compiles_cuda_only_sources(self):
        project = self.read("cuda/0BS_cuda.vcxproj")

        self.assertIn('<ProjectConfiguration Include="CUDA|x64">', project)
        self.assertIn("USE_CUDA", project)
        self.assertIn("CUDA 13.2.props", project)
        self.assertIn("CUDA 13.2.targets", project)
        self.assertIn('<CudaCompile Include="..\\detector\\cuda_preprocess.cu"', project)
        self.assertIn('<ClCompile Include="..\\detector\\trt_detector.cpp"', project)
        self.assertIn('<ClCompile Include="..\\depth\\depth_anything_trt.cpp"', project)
        self.assertIn('<ClCompile Include="..\\depth\\depth_utils.cpp"', project)
        self.assertIn('<ClCompile Include="..\\mem\\gpu_resource_manager.cpp"', project)
        self.assertIn("..\\x64\\CUDA\\", project)
        self.assertIn("$(MSBuildProjectDirectory)\\include", project)

    def test_cuda_folder_documents_dml_separation(self):
        readme = self.read("cuda/README.md").lower()
        self.assertIn("dml", readme)
        self.assertIn("separate", readme)
        self.assertIn("0bs_cuda.vcxproj", readme)
        self.assertIn("build-cuda.ps1", readme)
        self.assertIn("executionpolicy bypass", readme)
        self.assertIn("nanosim", readme)
        self.assertIn("dynamic test environment", readme)

    def test_cuda_build_script_has_clear_dependency_checks(self):
        script = self.read("cuda/build-cuda.ps1")

        self.assertIn("TensorRTDir", script)
        self.assertIn("TensorRT headers were not found", script)
        self.assertIn("Test-Path", script)
        self.assertIn("TensorRT-*", script)
        self.assertIn("nvinfer_10.lib", script)

    def test_cuda_compatibility_headers_are_local_to_cuda_folder(self):
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "nvinf.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "trt_monitor.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "tensorrt" / "nvinf.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "tensorrt" / "trt_monitor.h").exists())

        shim = self.read("cuda/include/nvinf.h")
        self.assertIn("createOptimizationProfile()", shim)
        self.assertNotIn("unique_ptr<nvinfer1::IOptimizationProfile", shim)

    def test_cuda_detector_includes_filesystem_for_engine_paths(self):
        detector = self.read("detector/trt_detector.cpp")

        self.assertIn("#include <filesystem>", detector)

    def test_cuda_project_uses_existing_opencv_install(self):
        project = self.read("cuda/0BS_cuda.vcxproj")

        self.assertIn("..\\modules\\opencv\\build\\install\\include", project)
        self.assertIn("..\\modules\\opencv\\build\\install\\x64\\vc18\\lib", project)
        self.assertIn("..\\modules\\opencv\\build\\install\\x64\\vc18\\bin\\opencv_world4140.dll", project)
        self.assertIn("$(TensorRTDir)\\bin\\*.dll", project)
        self.assertNotIn("opencv.sln", project.lower())
        self.assertNotIn("opencv_world.vcxproj", project.lower())

    def test_cuda_config_accepts_trt_backend_name(self):
        config = self.read("config/config.cpp")

        self.assertIn('backend = "TRT"', config)
        self.assertIn('backend == "CUDA"', config)
        self.assertIn('backend != "TRT"', config)


if __name__ == "__main__":
    unittest.main()
