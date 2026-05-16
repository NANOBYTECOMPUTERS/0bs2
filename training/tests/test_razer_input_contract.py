import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class RazerInputContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_razer_backend_is_exposed_in_config_ui_and_build_files(self):
        self.assertTrue((REPO_ROOT / "mouse/rzctl.h").exists())
        self.assertTrue((REPO_ROOT / "mouse/rzctl.cpp").exists())

        draw_mouse = self.read("overlay/draw_mouse.cpp")
        self.assertIn('"RAZER"', draw_mouse)
        self.assertIn('config.input_method == "RAZER"', draw_mouse)

        config_cpp = self.read("config/config.cpp")
        self.assertIn("RAZER", config_cpp)

        vcxproj = self.read("0BS_box_2.vcxproj")
        self.assertIn("mouse\\rzctl.cpp", vcxproj)
        self.assertIn("mouse\\rzctl.h", vcxproj)
        self.assertTrue((REPO_ROOT / "modules/razer-controls/x64/Release/rzctl.dll").exists())
        self.assertNotIn('None Include="rzctl.dll"', vcxproj)
        self.assertIn('None Include="modules\\razer-controls\\x64\\Release\\rzctl.dll"', vcxproj)
        self.assertIn('<Link>rzctl.dll</Link>', vcxproj)
        self.assertIn('SourceFiles="modules\\razer-controls\\x64\\Release\\rzctl.dll"', vcxproj)
        self.assertIn('DestinationFiles="$(OutDir)rzctl.dll"', vcxproj)

        cmake_path = REPO_ROOT / "CMakeLists.txt"
        if cmake_path.exists():
            cmake = cmake_path.read_text(encoding="utf-8")
            self.assertIn("mouse/rzctl.cpp", cmake)
            self.assertIn("BOX_RZCTL_DLL", cmake)

    def test_razer_backend_routes_movement_and_clicks_through_mouse_thread(self):
        main_cpp = self.read("0BS_box_2.cpp")
        mouse_input = self.read("mouse/MouseInput.cpp")
        self.assertIn("RzctlMouse* razerControl", main_cpp)
        self.assertIn("CreateMouseInputDevice(config)", main_cpp)
        self.assertIn("activeMouseInputOwner->razer()", main_cpp)
        self.assertIn("setMouseInput(activeMouseInputOwner.get())", main_cpp)
        self.assertIn("class RazerMouseInput", mouse_input)
        self.assertIn('method == "RAZER"', mouse_input)

        main_h = self.read("0BS_box_2.h")
        self.assertIn("extern RzctlMouse* razerControl", main_h)

        mouse_h = self.read("mouse/mouse.h")
        self.assertIn("IMouseInput* mouseInput", mouse_h)
        self.assertIn("setMouseInput", mouse_h)

        mouse_cpp = self.read("mouse/mouse.cpp")
        self.assertIn("mouseInput->move", mouse_cpp)
        self.assertIn("mouseInput->leftDown", mouse_cpp)
        self.assertIn("mouseInput->leftUp", mouse_cpp)

    def test_razer_wrapper_prefers_new_status_exports_without_extra_hooks(self):
        header = self.read("mouse/rzctl.h")
        impl = self.read("mouse/rzctl.cpp")

        self.assertIn("MouseMoveStatusFn", header)
        self.assertIn("MouseClickStatusFn", header)
        self.assertIn('GetProcAddress(rzctl, "mouse_move_status")', impl)
        self.assertIn('GetProcAddress(rzctl, "mouse_click_status")', impl)
        self.assertIn('modules" / L"razer-controls" / L"x64" / L"Release" / L"rzctl.dll"', impl)
        self.assertIn("mouseMoveStatus(x, y, TRUE)", impl)
        self.assertIn("mouseClickStatus(downFlagForKey(key))", impl)
        self.assertNotIn("is_initialized", impl)
        self.assertNotIn("get_last_error_code", impl)
        self.assertNotIn("get_failed_send_count", impl)
        self.assertNotIn("shutdown", impl)


if __name__ == "__main__":
    unittest.main()
