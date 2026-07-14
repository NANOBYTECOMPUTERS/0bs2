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
        self.assertNotIn("de" + "pth_anything_trt.cpp", project)
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
        self.assertIn("CudaBuildCustomizationVersion", project)
        self.assertIn("13.3", project)
        self.assertIn("CUDA $(CudaBuildCustomizationVersion).props", project)
        self.assertIn("CUDA $(CudaBuildCustomizationVersion).targets", project)
        self.assertIn('<CudaCompile Include="..\\detector\\cuda_preprocess.cu"', project)
        self.assertIn('<ClCompile Include="..\\detector\\trt_detector.cpp"', project)
        self.assertNotIn("..\\" + "de" + "pth\\", project)
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
        deps = self.read("cuda/build-deps.ps1")

        self.assertIn("TensorRTDir", script)
        self.assertIn("OpenCVDir", script)
        self.assertIn("Resolve-TensorRTDependency", script)
        self.assertIn("Resolve-OpenCVDependency", script)
        self.assertIn("TensorRTCoreLib", script)
        self.assertIn("TensorRTOnnxParserLib", script)
        self.assertIn("TensorRTPluginLib", script)
        self.assertNotIn("%3B", script)
        self.assertIn("TensorRT headers/libraries were not found", deps)
        self.assertIn("OpenCV CUDA install was not found", deps)
        self.assertIn("Test-Path", deps)
        self.assertIn("TensorRT-*", deps)
        self.assertIn("nvinfer_*.lib", deps)
        self.assertIn("CoreLib", deps)
        self.assertIn("OnnxParserLib", deps)
        self.assertIn("PluginLib", deps)
        self.assertIn("opencv_world*.lib", deps)
        self.assertIn("HAVE_CUDA", deps)
        self.assertIn("vswhere.exe", script)
        self.assertIn("Get-VisualStudioInstallRoots", script)
        self.assertIn("Resolve-MSBuild", script)
        self.assertIn("Resolve-CudaProps", script)
        self.assertIn("Resolve-CudaToolkitDir", script)
        self.assertIn("CudaToolkitDir", script)
        self.assertIn("CudaBuildCustomizationVersion", script)
        self.assertIn("VCTargetsPath", script)
        self.assertIn("$vcTargetsPath = Split-Path", script)
        self.assertIn("$vcTargetsPathForMsBuild = ($vcTargetsPath -replace '\\\\', '/')", script)
        self.assertIn("$vcTargetsPathForMsBuild += '/'", script)
        self.assertIn("$vcTargetsPathForMsBuild", script)
        self.assertIn("/p:VCTargetsPath=$vcTargetsPathForMsBuild", script)
        self.assertNotIn(r"Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe", script)
        self.assertNotIn(r"Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations", script)

    def test_cuda_compatibility_headers_are_local_to_cuda_folder(self):
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "nvinf.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "trt_monitor.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "trt_compat.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "tensorrt" / "nvinf.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "tensorrt" / "trt_monitor.h").exists())
        self.assertTrue((REPO_ROOT / "cuda" / "include" / "tensorrt" / "trt_compat.h").exists())

        shim = self.read("cuda/include/nvinf.h")
        compat = self.read("cuda/include/trt_compat.h")
        self.assertIn("createOptimizationProfile()", shim)
        self.assertNotIn("unique_ptr<nvinfer1::IOptimizationProfile", shim)
        self.assertIn("NV_TENSORRT_MAJOR < 11", compat)
        self.assertIn("explicitBatchNetworkFlags", compat)
        self.assertIn("enableFp16IfAvailable", compat)

    def test_tensorrt_11_removed_builder_flags_are_isolated_to_compat_header(self):
        compat = self.read("cuda/include/trt_compat.h")
        self.assertIn("kEXPLICIT_BATCH", compat)
        self.assertIn("kFP16", compat)

        for relative in ("cuda/include/nvinf.h", "detector/trt_detector.cpp"):
            text = self.read(relative)
            self.assertNotIn("NetworkDefinitionCreationFlag::kEXPLICIT_BATCH", text)
            self.assertNotIn("BuilderFlag::kFP16", text)
            self.assertNotIn("platformHasFastFp16", text)

    def test_cuda_detector_includes_filesystem_for_engine_paths(self):
        detector = self.read("detector/trt_detector.cpp")

        self.assertIn("#include <filesystem>", detector)

    def test_cuda_project_uses_existing_opencv_install(self):
        project = self.read("cuda/0BS_cuda.vcxproj")

        self.assertIn("..\\modules\\opencv-5.0.0\\build\\cuda\\install", project)
        self.assertIn("$(OpenCVDir)\\include", project)
        self.assertIn("$(OpenCVDir)\\x64\\vc18\\lib", project)
        self.assertIn("$(OpenCVWorldLib)", project)
        self.assertIn("opencv_world500.lib", project)
        self.assertIn("$(OpenCVDir)\\x64\\vc18\\bin\\opencv_*.dll", project)
        self.assertIn("$(TensorRTDir)\\bin\\*.dll", project)
        self.assertNotIn("opencv.sln", project.lower())
        self.assertNotIn("opencv_world.vcxproj", project.lower())

    def test_cuda_projects_default_to_tensorrt_11(self):
        project = self.read("cuda/0BS_cuda.vcxproj")
        worker = self.read("cuda/yolo_annotation_worker.vcxproj")

        for text in (project, worker):
            self.assertIn("TensorRT-11.1.0.106", text)
            self.assertIn("nvinfer_11.lib", text)
            self.assertIn("nvonnxparser_11.lib", text)
            self.assertIn("nvinfer_plugin_11.lib", text)
            self.assertNotIn("nvinfer_10.lib", text)
            self.assertNotIn("opencv_world4140.lib", text)

    def test_desktop_projects_link_static_glfw_for_static_runtime(self):
        main_project = self.read("0BS_box_2.vcxproj")
        cuda_project = self.read("cuda/0BS_cuda.vcxproj")

        for text in (main_project, cuda_project):
            self.assertIn("<RuntimeLibrary>MultiThreaded</RuntimeLibrary>", text)
            self.assertIn("glfw3_mt.lib", text)
            self.assertNotIn("glfw3dll.lib", text)

    def test_cuda_config_accepts_trt_backend_name(self):
        config = self.read("config/config.cpp")

        self.assertIn('backend = "TRT"', config)
        self.assertIn('backend == "CUDA"', config)
        self.assertIn('backend != "TRT"', config)

    def test_opencv_5_headers_are_explicit_where_transitive_includes_changed(self):
        main_cpp = self.read("0BS_box_2.cpp")
        overlay_loop = self.read("runtime/game_overlay_loop.cpp")

        self.assertIn("#include <opencv2/core/utils/logger.hpp>", main_cpp)
        self.assertIn("#include <opencv2/geometry/2d.hpp>", overlay_loop)

    def test_cuda_preprocess_avoids_opencv_cudev_convert_to(self):
        detector = self.read("detector/trt_detector.cpp")
        worker = self.read("worker/tensorrt_batch_detector.cpp")
        kernel = self.read("detector/cuda_preprocess.cu")

        for text in (detector, worker):
            self.assertNotIn(".convertTo(", text)
            self.assertIn("launch_hwc_to_chw_norm", text)

        self.assertIn("bgr8_to_rgb_chw_norm_kernel", kernel)
        self.assertIn("CV_8UC3", kernel)
        self.assertIn("static_cast<float>(p[2])", kernel)


if __name__ == "__main__":
    unittest.main()
