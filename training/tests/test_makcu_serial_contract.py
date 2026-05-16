import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class MakcuSerialContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_binary_baud_rate_packet_is_written_without_text_command_suffix(self):
        serialport = self.read("modules/makcu/src/serialport.cpp")
        match = re.search(
            r"bool\s+SerialPort::write\s*\(\s*const\s+std::vector<uint8_t>&\s+data\s*\)\s*"
            r"\{(?P<body>.*?)\n\s+bool\s+SerialPort::write\s*\(\s*const\s+std::string&\s+data\s*\)",
            serialport,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        body = match.group("body")

        self.assertIn("platformWrite", body)
        self.assertIn("data.data()", body)
        self.assertIn("data.size()", body)
        self.assertNotIn("sendCommand", body)
        self.assertNotIn("std::string(data.begin(), data.end())", body)

    def test_makcu_connection_normalizes_auto_ports_and_enables_buttons_after_connect(self):
        makcu_cpp = self.read("mouse/Makcu.cpp")

        self.assertIn("normalizeMakcuPort", makcu_cpp)
        self.assertIn('port == "COM0"', makcu_cpp)
        self.assertIn('port == "AUTO"', makcu_cpp)
        self.assertIn("device_.connect(normalized_port)", makcu_cpp)

        connect_pos = makcu_cpp.find("device_.connect(normalized_port)")
        enable_pos = makcu_cpp.find("device_.enableButtonMonitoring(true)", connect_pos)
        self.assertNotEqual(connect_pos, -1)
        self.assertNotEqual(enable_pos, -1)
        self.assertGreater(enable_pos, connect_pos)

    def test_makcu_commands_propagate_write_failures_to_mouse_input_fallback(self):
        makcu_h = self.read("mouse/Makcu.h")
        makcu_cpp = self.read("mouse/Makcu.cpp")
        mouse_input = self.read("mouse/MouseInput.cpp")

        self.assertIn("bool click(int button)", makcu_h)
        self.assertIn("bool press(int button)", makcu_h)
        self.assertIn("bool release(int button)", makcu_h)
        self.assertIn("bool move(int x, int y)", makcu_h)

        self.assertIn("device_.mouseMove(x, y)", makcu_cpp)
        self.assertIn("device_.click(toMakcuButton(button))", makcu_cpp)
        self.assertIn("device_.mouseDown(toMakcuButton(button))", makcu_cpp)
        self.assertIn("device_.mouseUp(toMakcuButton(button))", makcu_cpp)
        self.assertIn("is_open_ = false", makcu_cpp)
        self.assertIn("return sendWin32Move(dx, dy);", mouse_input)
        self.assertIn("if (device_->move(dx, dy))", mouse_input)
        self.assertIn("if (device_->press(0))", mouse_input)
        self.assertIn("if (device_->release(0))", mouse_input)

    def test_makcu_default_config_and_ui_offer_auto_port(self):
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")

        self.assertIn('makcu_port = "AUTO"', config_cpp)
        self.assertIn('get_string("makcu_port", "AUTO")', config_cpp)
        self.assertIn("makcu_baudrate = 4000000", config_cpp)
        self.assertIn('get_long("makcu_baudrate", 4000000)', config_cpp)
        self.assertIn('ports.push_back("AUTO")', draw_mouse)
        self.assertIn("4000000", draw_mouse)

        runtime_config = self.read("x64/DML/config.ini")
        dist_config = self.read("dist/0BS/config.ini")
        self.assertIn("makcu_port = AUTO", runtime_config)
        self.assertIn("makcu_port = AUTO", dist_config)

    def test_parallel_build_serial_units_use_synchronized_pdb_writes(self):
        vcxproj = self.read("0BS_box_2.vcxproj")

        self.assertIn("<AdditionalOptions>/FS %(AdditionalOptions)</AdditionalOptions>", vcxproj)


if __name__ == "__main__":
    unittest.main()
