import unittest
import re
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "MouseInput.cpp").exists()
)


class Teensy41MouseContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_teensy41_is_a_distinct_serial_input_method(self):
        mouse_h = self.read("mouse/MouseInput.h")
        mouse_cpp = self.read("mouse/MouseInput.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        config_cpp = self.read("config/config.cpp")

        self.assertIn("Teensy41", mouse_h)
        self.assertIn('method == "TEENSY41"', mouse_cpp)
        self.assertIn('return "TEENSY41"', mouse_cpp)
        self.assertIn("ArduinoProtocol::Teensy41", mouse_cpp)
        self.assertIn("Teensy41MouseInput", mouse_cpp)
        self.assertIn('"TEENSY41"', draw_mouse)
        self.assertIn('config.input_method == "TEENSY41"', draw_mouse)
        self.assertIn("TEENSY41", config_cpp)

    def test_teensy41_protocol_matches_firmware_commands(self):
        arduino_h = self.read("mouse/Arduino.h")
        arduino_cpp = self.read("mouse/Arduino.cpp")

        self.assertIn("enum class ArduinoProtocol", arduino_h)
        self.assertIn("Teensy41", arduino_h)
        self.assertIn("protocol_ == ArduinoProtocol::Teensy41", arduino_cpp)
        self.assertIn('"move " + std::to_string(x)', arduino_cpp)
        self.assertIn('" 0 0\\n"', arduino_cpp)
        self.assertIn('"buttons " + std::to_string(button_mask_)', arduino_cpp)
        self.assertIn("startListening()", arduino_cpp)
        self.assertIn("zooming_active = pressed", arduino_cpp)
        self.assertIn("zooming.store(pressed)", arduino_cpp)

    def test_teensy41_serial_mouse_input_does_not_fallback_to_win32_when_unplugged(self):
        mouse_cpp = self.read("mouse/MouseInput.cpp")

        teensy_class = re.search(
            r"class\s+Teensy41MouseInput\b(?P<body>.*?)\n\s*private:",
            mouse_cpp,
            re.DOTALL,
        )
        self.assertIsNotNone(teensy_class)
        body = teensy_class.group("body")
        self.assertNotIn("allowWin32Fallback", body)
        self.assertNotIn("return sendWin32Move", body)
        self.assertNotIn("return sendWin32Click", body)

    def test_teensy41_button_state_bypasses_window_focus(self):
        keyboard_cpp = self.read("keyboard/keyboard_listener.cpp")

        self.assertIn("isTeensy41InputMethod()", keyboard_cpp)
        self.assertIn('config.input_method == "TEENSY41"', keyboard_cpp)
        self.assertIn("useArduinoButtonState", keyboard_cpp)
        self.assertIn("config.arduino_enable_keys || isTeensy41InputMethod()", keyboard_cpp)
        self.assertIn("useArduinoButtonState && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->aiming_active", keyboard_cpp)
        self.assertIn("useArduinoButtonState && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->shooting_active", keyboard_cpp)
        self.assertIn("useArduinoButtonState && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->zooming_active", keyboard_cpp)

    def test_teensy41_reference_firmware_supports_buttons_without_move_echo(self):
        self.assert_teensy41_sketch_supports_buttons_without_move_echo(
            "firmware/teensy41_mouse_bridge/teensy41_mouse_bridge.ino"
        )

    def test_teensy41_user_passthrough_sketch_supports_buttons_without_move_echo(self):
        self.assert_teensy41_sketch_supports_buttons_without_move_echo(
            "TeensyMouseSerialPassthrough/TeensyMouseSerialPassthrough.ino"
        )

    def assert_teensy41_sketch_supports_buttons_without_move_echo(self, relative_path):
        sketch = self.read(relative_path)

        self.assertIn('sscanf(line, "buttons %d"', sketch)
        self.assertIn("applyButtons()", sketch)
        self.assertIn("hostButtons | serialButtons", sketch)
        self.assertIn("emitButtonTransitions(hostButtons, buttons)", sketch)
        self.assertIn('Serial.printf("BD:%u\\n"', sketch)
        self.assertIn('Serial.printf("BU:%u\\n"', sketch)
        move_branch = sketch.split('sscanf(line, "move %d %d %d %d"')[1].split('sscanf(line, "buttons %d"')[0]
        self.assertNotIn('Serial.println("ok")', move_branch)


if __name__ == "__main__":
    unittest.main()
