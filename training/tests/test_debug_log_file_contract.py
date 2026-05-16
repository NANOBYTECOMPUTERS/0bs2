import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class DebugLogFileContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_config_exposes_optional_log_file_settings(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        self.assertRegex(config_h, r"\bbool\s+debug_log_file_enabled\s*;")
        self.assertRegex(config_h, r"\bstd::string\s+debug_log_file_path\s*;")

        self.assertIn("debug_log_file_enabled = false;", config_cpp)
        self.assertIn('debug_log_file_path = "logs/0BS.log";', config_cpp)
        self.assertIn('debug_log_file_enabled = get_bool("debug_log_file_enabled", false);', config_cpp)
        self.assertIn('debug_log_file_path = get_string("debug_log_file_path", "logs/0BS.log");', config_cpp)
        self.assertIn('"debug_log_file_enabled = " << (debug_log_file_enabled ? "true" : "false")', config_cpp)
        self.assertIn('"debug_log_file_path = " << debug_log_file_path', config_cpp)
        self.assertIn('if (debug_log_file_path.empty()) debug_log_file_path = "logs/0BS.log";', config_cpp)

    def test_debug_tab_exposes_file_logging_controls(self):
        debug_gui = self.read("overlay/draw_debug.cpp")

        self.assertIn("debugLogFilePathBuf", debug_gui)
        self.assertIn("prev_debug_log_file_enabled", debug_gui)
        self.assertIn("config.debug_log_file_enabled", debug_gui)
        self.assertIn("config.debug_log_file_path", debug_gui)
        self.assertIn('ImGui::Checkbox("Enable log file"', debug_gui)
        self.assertIn('ImGui::InputText("Log file path"', debug_gui)
        self.assertIn("OverlayConfig_MarkDirty()", debug_gui)

    def test_standalone_logger_files_define_optional_file_logger(self):
        header_path = REPO_ROOT / "diagnostics" / "Logger.h"
        impl_path = REPO_ROOT / "diagnostics" / "Logger.cpp"
        self.assertTrue(header_path.exists(), "diagnostics/Logger.h should exist")
        self.assertTrue(impl_path.exists(), "diagnostics/Logger.cpp should exist")

        header = header_path.read_text(encoding="utf-8")
        impl = impl_path.read_text(encoding="utf-8")

        self.assertIn("namespace diagnostics", header)
        self.assertIn("class Logger", header)
        self.assertIn("configure", header)
        self.assertIn("setFileLogging", header)
        self.assertIn("log", header)
        self.assertIn("isFileLoggingEnabled", header)
        self.assertIn("logFilePath", header)
        self.assertIn('std::string log_file_path_ = "logs/0BS.log";', header)
        self.assertIn("bool file_logging_enabled_ = false;", header)

        self.assertIn("std::filesystem::create_directories", impl)
        self.assertIn("std::ofstream file", impl)
        self.assertRegex(impl, re.compile(r"if\s*\(\s*!file_logging_enabled_\s*\|\|\s*log_file_path_\.empty\(\)\s*\)"))
        self.assertIn("std::ios::app", impl)

    def test_logger_is_wired_into_visual_studio_project(self):
        project = self.read("0BS_box_2.vcxproj")

        self.assertIn("diagnostics\\Logger.cpp", project)
        self.assertIn("diagnostics\\Logger.h", project)


if __name__ == "__main__":
    unittest.main()
