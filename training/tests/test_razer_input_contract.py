import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "MouseInput.cpp").exists()
)


class RazerInputContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_razer_backend_is_exposed_in_config_ui_and_build_files(self):
        self.assertTrue((REPO_ROOT / "mouse/rzctl.h").exists())
        self.assertTrue((REPO_ROOT / "mouse/rzctl.cpp").exists())

        draw_mouse = self.read("overlay/draw_mouse.cpp")
        self.assertIn('"RAZER"', draw_mouse)
        self.assertIn('config.input_method == "RAZER"', draw_mouse)
        # After inlining: the RAZER UI branch must mention the in-process / direct nature
        # and must no longer advertise an external rzctl.dll requirement.
        self.assertIn("in-process", draw_mouse)

        config_cpp = self.read("config/config.cpp")
        self.assertIn("RAZER", config_cpp)

        vcxproj = self.read("0BS_box_2.vcxproj")
        self.assertIn("mouse\\rzctl.cpp", vcxproj)
        self.assertIn("mouse\\rzctl.h", vcxproj)
        # The external DLL must no longer be referenced as a required/None item or copy target.
        self.assertNotIn('None Include="rzctl.dll"', vcxproj)
        self.assertNotIn('razer-controls', vcxproj)  # no more subproject DLL reference in main project
        self.assertNotIn('CopyRazerControlsDll', vcxproj)

        # The legacy modules folder may still exist for reference, but is not a build dep.
        # We do not assert its absence (it can remain as docs), but the main build must not require it.

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

    def test_razer_lazy_open_is_not_blocked_by_wrapper_is_open_gate(self):
        mouse_input = self.read("mouse/MouseInput.cpp")
        start = mouse_input.index("class RazerMouseInput")
        end = mouse_input.index("class DirectMouseInput", start)
        razer_block = mouse_input[start:end]

        self.assertIn("device_->mouse_xy(dx, dy)", razer_block)
        self.assertIn("device_->mouse_down()", razer_block)
        self.assertIn("device_->mouse_up()", razer_block)
        self.assertNotIn("return isOpen() && device_->mouse_xy", razer_block)
        self.assertNotIn("return isOpen() && device_->mouse_down", razer_block)
        self.assertNotIn("return isOpen() && device_->mouse_up", razer_block)

    def test_razer_is_now_direct_in_process_no_external_loader(self):
        """After the stealth inlining the RAZER backend must be self-contained.
        No LoadLibrary of rzctl.dll, no GetProcAddress for the old exports,
        no references to the old modules path inside the implementation.
        """
        impl = self.read("mouse/rzctl.cpp")
        header = self.read("mouse/rzctl.h")

        # Must not contain the old *rzctl DLL* loader patterns.
        # Dynamic GetProcAddress for *ntdll* (stealthy NT resolution) is expected and desired.
        self.assertNotIn("LoadLibrary", impl)
        self.assertNotIn("rzctl.dll", impl)
        self.assertNotIn("razer-controls", impl)
        # Old rzctl-specific GetProcAddress for the helper DLL exports must be gone.
        self.assertNotIn('GetProcAddress(rzctl, "mouse_move_status")', impl)
        self.assertNotIn('GetProcAddress(rzctl, "init")', impl)

        # Must contain evidence of direct (inlined) implementation.
        # The new code uses DeviceIoControl + the MOUSE_IOCTL_STRUCT (or equivalent direct path).
        hasDirectEvidence = (
            ("DeviceIoControl" in impl) or
            ("MOUSE_IOCTL_STRUCT" in impl) or
            ("GetIoctlMouse" in impl) or
            ("InitDeviceUnlocked" in impl) or
            ("FindSymLinkInternal" in impl)
        )
        self.assertTrue(hasDirectEvidence, "Expected direct RZCONTROL/DeviceIoControl logic in the inlined rzctl.cpp")

        # Header should be minimal (no loader types)
        self.assertNotIn("InitFn", header)
        self.assertNotIn("HMODULE rzctl", header)

    def test_razer_inlined_device_handle_is_closed_on_mouse_close(self):
        impl = self.read("mouse/rzctl.cpp")

        self.assertIn("ShutdownDevice", impl)
        self.assertIn("RzctlMouse::~RzctlMouse()", impl)
        self.assertIn("mouse_close();", impl)
        self.assertIn("bool RzctlMouse::mouse_close()", impl)
        self.assertIn("detail::ShutdownDevice()", impl)

    def test_razer_symlink_discovery_enumerates_until_match(self):
        impl = self.read("mouse/rzctl.cpp")
        start = impl.index("static bool FindSymLinkInternal")
        end = impl.index("static bool InitDeviceUnlocked", start)
        find_block = impl[start:end]

        self.assertIn("while (true)", find_block)
        self.assertIn("queryContext", find_block)
        self.assertIn("STATUS_BUFFER_TOO_SMALL", find_block)
        self.assertIn("name.find(contains)", find_block)
        self.assertIn("break;", find_block)


if __name__ == "__main__":
    unittest.main()
