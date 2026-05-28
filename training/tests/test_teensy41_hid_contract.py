import re
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "MouseInput.cpp").exists()
)


class Teensy41HidContractTest(unittest.TestCase):
    def read(self, relative_path):
        path = REPO_ROOT / relative_path
        self.assertTrue(path.exists(), f"Expected source file to exist: {relative_path}")
        return path.read_text(encoding="utf-8")

    def assert_contains_all(self, text, needles):
        missing = [needle for needle in needles if needle not in text]
        self.assertFalse(missing, f"Missing expected text: {missing}")

    def assert_contains(self, text, needle):
        if needle not in text:
            self.fail(f"Missing expected text: {needle}")

    def assert_matches(self, text, pattern):
        self.assertIsNotNone(
            re.search(pattern, text, re.DOTALL),
            f"Missing expected pattern: {pattern}",
        )

    def assert_not_contains(self, text, needle):
        if needle in text:
            self.fail(f"Unexpected text present: {needle}")

    def assert_not_matches(self, text, pattern):
        self.assertIsNone(
            re.search(pattern, text, re.DOTALL),
            f"Unexpected pattern present: {pattern}",
        )

    def assert_contains_all_legacy(self, text, needles):
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_teensy41_hid_is_a_distinct_mouse_input_method(self):
        mouse_h = self.read("mouse/MouseInput.h")
        mouse_cpp = self.read("mouse/MouseInput.cpp")

        self.assert_contains(mouse_h, "Teensy41Hid")
        self.assert_contains(mouse_cpp, 'method == "TEENSY41_HID"')
        self.assert_contains(mouse_cpp, 'return "TEENSY41_HID"')
        self.assert_matches(mouse_cpp, r"class\s+Teensy41(?:RawHid|Hid)MouseInput\b")
        self.assert_matches(mouse_cpp, r"std::make_unique<Teensy41(?:RawHid|Hid)MouseInput>")

        hid_factory_match = re.search(
            r"case\s+MouseInputMethod::Teensy41Hid\s*:(?P<body>.*?)\n\s*case\s+MouseInputMethod::",
            mouse_cpp,
            re.DOTALL,
        )
        self.assertIsNotNone(hid_factory_match)
        self.assert_not_contains(hid_factory_match.group("body"), "ArduinoProtocol::Teensy41")
        self.assert_not_contains(hid_factory_match.group("body"), "config.arduino_port")
        self.assert_not_contains(hid_factory_match.group("body"), "config.arduino_baudrate")

    def test_config_exposes_hid_identity_filters_and_timing_defaults(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        fields = [
            "teensy_hid_manufacturer",
            "teensy_hid_product",
            "teensy_hid_serial",
            "teensy_hid_vid_filter",
            "teensy_hid_pid_filter",
            "teensy_hid_usage_page",
            "teensy_hid_usage_id",
            "teensy_hid_open_index",
            "teensy_hid_packet_timeout_ms",
            "teensy_hid_reconnect_interval_ms",
        ]
        self.assert_contains_all(config_h, fields)
        self.assert_contains_all(config_cpp, fields)

        expected_defaults = {
            "teensy_hid_manufacturer": '"Generic"',
            "teensy_hid_product": '"USB HID Mouse"',
            "teensy_hid_serial": '"AUTO"',
            "teensy_hid_vid_filter": '"AUTO"',
            "teensy_hid_pid_filter": '"AUTO"',
            "teensy_hid_usage_page": r"(?:0xFFAB|65451)",
            "teensy_hid_usage_id": r"(?:0x0200|512)",
            "teensy_hid_open_index": "0",
            "teensy_hid_packet_timeout_ms": "2",
            "teensy_hid_reconnect_interval_ms": "500",
        }
        for field, default_pattern in expected_defaults.items():
            with self.subTest(field=field):
                self.assert_matches(
                    config_cpp,
                    rf'{field}\s*=\s*{default_pattern}|"{field}"\s*,\s*{default_pattern}',
                )

    def test_rawhid_packet_contract_lives_in_mouse_header(self):
        header = self.read("mouse/Teensy41RawHid.h")

        self.assert_matches(header, r"struct\s+Teensy41(?:RawHid)?(?:Move)?Packet\b")
        self.assert_matches(header, r"static_assert\s*\(\s*sizeof\s*\([^)]*Packet[^)]*\)\s*==\s*64\b")
        self.assert_contains(header, "magic")
        self.assert_contains(header, "version")
        self.assert_matches(header, r"\bint16_t\s+dx\b")
        self.assert_matches(header, r"\bint16_t\s+dy\b")
        self.assert_matches(header, r"\bint16_t\s+wheel\b")
        self.assert_matches(header, r"\bint16_t\s+wheelH\b")
        self.assert_matches(header, r"\buint8_t\s+buttonMask\b")

    def test_rawhid_mouse_input_does_not_fallback_to_win32_when_unplugged(self):
        mouse_h = self.read("mouse/MouseInput.h")
        mouse_cpp = self.read("mouse/MouseInput.cpp")
        runtime_mouse = self.read("mouse/mouse.cpp")

        self.assert_not_contains(mouse_h, "allowWin32Fallback")
        hid_class = re.search(
            r"class\s+Teensy41RawHidMouseInput\b(?P<body>.*?)\n\s*private:",
            mouse_cpp,
            re.DOTALL,
        )
        self.assertIsNotNone(hid_class)
        body = hid_class.group("body")
        self.assert_not_contains(body, "allowWin32Fallback")
        self.assert_not_contains(body, "return sendWin32Move")
        self.assert_not_contains(body, "return sendWin32Click")
        self.assert_not_contains(runtime_mouse, "SendInput")

    def test_button_state_methods_are_available_to_keyboard_listener(self):
        mouse_h = self.read("mouse/MouseInput.h")
        keyboard_cpp = self.read("keyboard/keyboard_listener.cpp")

        self.assert_matches(mouse_h, r"virtual\s+bool\s+aimingActive\s*\(\s*\)")
        self.assert_matches(mouse_h, r"virtual\s+bool\s+shootingActive\s*\(\s*\)")
        self.assert_matches(mouse_h, r"virtual\s+bool\s+zoomingActive\s*\(\s*\)")

        self.assert_contains(keyboard_cpp, 'config.input_method == "TEENSY41_HID"')
        self.assert_contains(keyboard_cpp, "activeMouseInputOwner")
        self.assert_contains(keyboard_cpp, "aimingActive()")
        self.assert_contains(keyboard_cpp, "shootingActive()")
        self.assert_contains(keyboard_cpp, "zoomingActive()")
        self.assert_contains(keyboard_cpp, "return isTeensy41InputMethod() &&")
        self.assert_not_contains(keyboard_cpp, "return (config.arduino_enable_keys || isTeensy41InputMethod()) &&")

    def test_gui_lists_hid_and_uses_hid_settings_instead_of_serial_settings(self):
        draw_mouse = self.read("overlay/draw_mouse.cpp")

        self.assert_contains(draw_mouse, '"TEENSY41_HID"')
        self.assert_contains(draw_mouse, 'config.input_method == "TEENSY41_HID"')
        self.assert_contains_all(
            draw_mouse,
            [
                "teensy_hid_manufacturer",
                "teensy_hid_product",
                "teensy_hid_serial",
                "teensy_hid_vid_filter",
                "teensy_hid_pid_filter",
                "teensy_hid_usage_page",
                "teensy_hid_usage_id",
                "teensy_hid_open_index",
                "teensy_hid_packet_timeout_ms",
                "teensy_hid_reconnect_interval_ms",
                "Reset HID defaults",
            ],
        )

        hid_branch = re.search(
            r'config\.input_method\s*==\s*"TEENSY41_HID".*?(?=\n\s*else\s+if\s*\(|\n\s*else\s*\{)',
            draw_mouse,
            re.DOTALL,
        )
        self.assertIsNotNone(hid_branch)
        self.assert_not_contains(hid_branch.group(0), "arduino_port")
        self.assert_not_contains(hid_branch.group(0), "arduino_baudrate")

    def test_rawhid_firmware_uses_usb_rawhid_packets_not_serial_move_text(self):
        sketch = self.read("TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino")

        self.assert_contains(sketch, "RawHID.recv")
        self.assert_contains(sketch, "RawHID.send")
        self.assert_contains(sketch, "Teensy41RawHid")
        self.assert_matches(sketch, r"__attribute__\s*\(\(\s*packed\s*\)\)\s*RawHidHostPacket")
        self.assert_matches(sketch, r"__attribute__\s*\(\(\s*packed\s*\)\)\s*RawHidDeviceEventPacket")
        self.assert_not_contains(sketch, "Serial.available")
        self.assert_not_contains(sketch, "sscanf")
        self.assert_not_contains(sketch, '"move %d %d')
        self.assert_not_contains(sketch, "usb_set_")

    def test_rawhid_firmware_uses_compile_time_usb_name_overrides(self):
        name_c = self.read("TeensyMouseRawHidBridge/name.c")

        self.assert_contains(name_c, '#include "usb_names.h"')
        self.assert_contains(name_c, "usb_string_manufacturer_name")
        self.assert_contains(name_c, "usb_string_product_name")
        self.assert_contains(name_c, "MANUFACTURER_NAME_LEN 7")
        self.assert_contains(name_c, "PRODUCT_NAME_LEN 13")
        self.assert_contains(name_c, "'G','e','n','e','r','i','c'")
        self.assert_contains(name_c, "'U','S','B',' ','H','I','D',' ','M','o','u','s','e'")


if __name__ == "__main__":
    unittest.main()
