import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


def join(*parts: str) -> str:
    return "".join(parts)


class RemovedMlCleanupContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_removed_runtime_files_stay_removed(self):
        removed = [
            Path(join("ne", "ural")),
            Path(join("de", "pth")),
            Path(join("ne", "ural", "_models")),
            Path("include") / join("real", "_world", "_data", "_logger.h"),
            Path("mouse") / join("Pid", "Gover", "nor.cpp"),
            Path("mouse") / join("Pid", "Gover", "nor.h"),
            Path("mouse") / join("Pid", "Mouse", "Controller.cpp"),
            Path("mouse") / join("Pid", "Mouse", "Controller.h"),
            Path("mouse") / join("Smart", "Blender.cpp"),
            Path("mouse") / join("Smart", "Blender.h"),
            Path("mouse") / join("Onnx", "Pid", "Gover", "nor.cpp"),
            Path("mouse") / join("Onnx", "Pid", "Gover", "nor.h"),
            Path("overlay") / join("draw_", "ne", "ural.cpp"),
            Path("overlay") / join("draw_", "de", "pth.cpp"),
        ]

        for relative in removed:
            self.assertFalse((REPO_ROOT / relative).exists(), str(relative))

    def test_project_files_do_not_reference_removed_sources(self):
        project_text = "\n".join(
            self.read(path)
            for path in (
                "0BS_box_2.vcxproj",
                "0BS_box_2.vcxproj.filters",
                "cuda/0BS_cuda.vcxproj",
                "cuda/0BS_cuda.vcxproj.filters",
            )
        )
        banned = [
            join("ne", "ural", "\\"),
            join("ne", "ural", "/"),
            join("de", "pth", "\\"),
            join("de", "pth", "/"),
            join("Pid", "Gover", "nor"),
            join("Pid", "Mouse", "Controller"),
            join("Smart", "Blender"),
            join("draw_", "ne", "ural"),
            join("draw_", "de", "pth"),
        ]

        for token in banned:
            self.assertNotIn(token, project_text)

    def test_config_and_runtime_do_not_expose_removed_keys(self):
        text = "\n".join(
            self.read(path)
            for path in (
                "config/config.h",
                "config/config.cpp",
                "overlay/draw_settings.h",
                "overlay/overlay.cpp",
                "overlay/draw_mouse.cpp",
                "runtime/mouse_thread_loop.cpp",
                "runtime/game_overlay_loop.cpp",
                "mouse/mouse.h",
                "mouse/mouse.cpp",
                "mouse/BoxTarget.h",
                "mouse/BoxTarget.cpp",
                "capture/capture.h",
                "capture/capture.cpp",
                "detector/dml_detector.cpp",
                "detector/trt_detector.cpp",
            )
        )
        banned = [
            join("pid_", "gover", "nor"),
            join("ne", "ural_"),
            join("temp", "oral_prediction"),
            join("adaptive_", "prediction"),
            join("base_", "prediction", "_influence"),
            join("prediction_", "lead", "_smoothing"),
            join("de", "pth_"),
            join("real_", "world"),
            join("Ne", "ural"),
            join("Temp", "oral", "Predictor"),
            join("Pid", "Gover", "nor"),
            join("Pid", "Mouse", "Controller"),
            join("Smart", "Blender"),
            join("feed", "_forward"),
            join("smart", "_blending"),
        ]

        for token in banned:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
