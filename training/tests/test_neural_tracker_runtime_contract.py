import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class NeuralTrackerRuntimeContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_config_persists_neural_tracker_runtime(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        self.assertIn("std::string neural_tracker_runtime", config_h)
        self.assertIn('neural_tracker_runtime = "CPU"', config_cpp)
        self.assertIn('get_string("neural_tracker_runtime", "CPU")', config_cpp)
        self.assertIn('MERGE_FIELD("neural_tracker_runtime", neural_tracker_runtime)', config_cpp)
        self.assertIn('"neural_tracker_runtime = " << neural_tracker_runtime', config_cpp)
        self.assertIn('neural_tracker_runtime != "CUDA"', config_cpp)

    def test_neural_tab_exposes_model_picker_and_runtime_selector(self):
        draw_neural = self.read("overlay/draw_neural.cpp")

        self.assertIn("getAvailableNeuralTrackerModels", draw_neural)
        self.assertIn('ImGui::Combo("Association model"', draw_neural)
        self.assertIn('ImGui::Combo("Association runtime"', draw_neural)
        self.assertIn('config.neural_tracker_runtime = "CUDA"', draw_neural)
        self.assertIn('config.neural_tracker_runtime = "CPU"', draw_neural)

    def test_tracker_factory_receives_runtime_explicitly(self):
        header = self.read("neural/NeuralTracker.h")
        impl = self.read("neural/NeuralTracker.cpp")
        box_target = self.read("mouse/BoxTarget.cpp")

        self.assertIn("createNeuralTracker(const std::string& modelPath, const std::string& runtime)", header)
        self.assertIn("createCpuNeuralTracker", impl)
        self.assertIn("createCudaNeuralTracker", impl)
        self.assertIn("config.neural_tracker_runtime", box_target)
        self.assertNotIn("createOnnxNeuralTracker(loadedPath)", box_target)

    def test_cuda_runtime_builds_or_loads_tensor_rt_engine(self):
        impl = self.read("neural/NeuralTracker.cpp")

        self.assertIn("#ifdef USE_CUDA", impl)
        self.assertIn("#include \"tensorrt/nvinf.h\"", impl)
        self.assertIn("class TrtNeuralTracker", impl)
        self.assertIn("resolveNeuralEnginePath", impl)
        self.assertIn("buildEngineFromOnnx", impl)
        self.assertIn("loadEngineFromFile", impl)
        self.assertIn("enqueueV3", impl)
        self.assertIn("cudaMemcpyAsync", impl)


if __name__ == "__main__":
    unittest.main()
