import unittest
import re
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "MouseInput.cpp").exists()
)


class RuntimeArchitectureContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_mouse_inputs_are_hidden_behind_shared_interface_and_factory(self):
        header_path = REPO_ROOT / "mouse" / "MouseInput.h"
        impl_path = REPO_ROOT / "mouse" / "MouseInput.cpp"
        self.assertTrue(header_path.exists())
        self.assertTrue(impl_path.exists())

        header = header_path.read_text(encoding="utf-8")
        impl = impl_path.read_text(encoding="utf-8")
        main_cpp = self.read("0BS_box_2.cpp")
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")

        self.assertIn("enum class MouseInputMethod", header)
        self.assertIn("class IMouseInput", header)
        self.assertIn("std::unique_ptr<IMouseInput> CreateMouseInputDevice", header)
        self.assertIn("class RazerMouseInput", impl)
        self.assertIn("class Win32MouseInput", impl)
        self.assertIn("std::unique_ptr<IMouseInput> activeMouseInputOwner", main_cpp)
        self.assertIn("CreateMouseInputDevice(config)", main_cpp)
        self.assertNotIn("new Arduino", main_cpp)
        self.assertNotIn("delete arduinoSerial", main_cpp)

        self.assertIn("IMouseInput* mouseInput", mouse_h)
        self.assertIn("setMouseInput(IMouseInput* newMouseInput)", mouse_h)
        self.assertNotIn("RzctlMouse* razer;", mouse_h)
        self.assertNotIn("GhubMouse* gHub;", mouse_h)
        self.assertIn("mouseInput->move(dx, dy)", mouse_cpp)
        self.assertIn("mouseInput->leftDown()", mouse_cpp)
        self.assertIn("mouseInput->leftUp()", mouse_cpp)

    def test_only_explicit_win32_method_uses_win32_mouse_packets(self):
        mouse_h = self.read("mouse/MouseInput.h")
        mouse_input = self.read("mouse/MouseInput.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")
        main_cpp = self.read("0BS_box_2.cpp")
        ghub_cpp = self.read("mouse/ghub.cpp")
        rzctl_cpp = self.read("mouse/rzctl.cpp")

        self.assertNotIn("allowWin32Fallback", mouse_h)
        self.assertNotIn("allowWin32Fallback", mouse_input)
        self.assertNotIn("allowWin32Fallback", mouse_cpp)
        self.assertNotIn("SendInput", mouse_cpp)
        self.assertNotIn("Win32 fallback remains available", main_cpp)
        self.assertNotIn("SendInput", ghub_cpp)
        self.assertNotIn("SendInput", rzctl_cpp)

        for class_name in [
            "ArduinoMouseInput",
            "Teensy41MouseInput",
            "Teensy41RawHidMouseInput",
            "GHubMouseInput",
            "RazerMouseInput",
            "KmboxNetMouseInput",
            "KmboxAMouseInput",
            "MakcuMouseInput",
        ]:
            with self.subTest(class_name=class_name):
                match = re.search(
                    rf"class\s+{class_name}\b(?P<body>.*?)\n\s*private:",
                    mouse_input,
                    re.DOTALL,
                )
                self.assertIsNotNone(match)
                body = match.group("body")
                self.assertNotIn("sendWin32Move", body)
                self.assertNotIn("sendWin32Click", body)

    def test_runtime_threads_are_owned_by_supervisor(self):
        header_path = REPO_ROOT / "runtime" / "RuntimeSupervisor.h"
        impl_path = REPO_ROOT / "runtime" / "RuntimeSupervisor.cpp"
        self.assertTrue(header_path.exists())
        self.assertTrue(impl_path.exists())

        header = header_path.read_text(encoding="utf-8")
        main_cpp = self.read("0BS_box_2.cpp")

        self.assertIn("class RuntimeSupervisor", header)
        self.assertIn("void joinAll()", header)
        self.assertIn("RuntimeSupervisor runtime", main_cpp)
        self.assertIn('runtime.start("KeyboardListener"', main_cpp)
        self.assertIn('runtime.start("CaptureThread"', main_cpp)
        self.assertIn("runtime.joinAll()", main_cpp)

    def test_config_has_typed_input_validation(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        self.assertIn("bool validate();", config_h)
        self.assertIn("ParseMouseInputMethod(input_method)", config_cpp)
        self.assertIn('input_method = "WIN32"', config_cpp)
        self.assertIn("validate();", config_cpp)

    def test_project_builds_new_runtime_architecture_files(self):
        vcxproj = self.read("0BS_box_2.vcxproj")

        self.assertIn('ClCompile Include="mouse\\MouseInput.cpp"', vcxproj)
        self.assertIn('ClInclude Include="mouse\\MouseInput.h"', vcxproj)
        self.assertIn('ClCompile Include="runtime\\RuntimeSupervisor.cpp"', vcxproj)
        self.assertIn('ClInclude Include="runtime\\RuntimeSupervisor.h"', vcxproj)

    def test_windows_mouse_headers_do_not_leak_min_max_macros(self):
        mouse_h = self.read("mouse/mouse.h")
        draw_mouse = self.read("overlay/draw_mouse.cpp")

        self.assertIn("#define NOMINMAX", mouse_h)
        self.assertLess(mouse_h.index("#define NOMINMAX"), mouse_h.index("#include <Windows.h>"))
        self.assertIn('#include "rzctl.h"', draw_mouse)

    def test_capture_submits_cached_frames_when_backend_has_no_new_desktop_frame(self):
        capture_cpp = self.read("capture/capture.cpp")

        self.assertIn("lastDetectionFrame", capture_cpp)
        self.assertIn("reusedCachedDetectionFrame", capture_cpp)
        self.assertIn("submitFrameToDetector", capture_cpp)
        self.assertIn("lastDetectionFrame = detectionFrame.clone()", capture_cpp)
        self.assertIn("detectionFrame = lastDetectionFrame", capture_cpp)


if __name__ == "__main__":
    unittest.main()
