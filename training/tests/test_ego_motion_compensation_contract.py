import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


def find_vsdev() -> Path | None:
    candidates: list[Path] = []
    env_vsdev = os.environ.get("VSDEVCMD")
    if env_vsdev:
        candidates.append(Path(env_vsdev))

    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if vswhere.exists():
        probe = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                r"Common7\Tools\VsDevCmd.bat",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        for line in probe.stdout.splitlines():
            if line.strip():
                candidates.append(Path(line.strip()))

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.extend(
        [
            program_files / "Microsoft Visual Studio" / "18" / "Community" / "Common7" / "Tools" / "VsDevCmd.bat",
            program_files / "Microsoft Visual Studio" / "18" / "BuildTools" / "Common7" / "Tools" / "VsDevCmd.bat",
            program_files / "Microsoft Visual Studio" / "2022" / "Community" / "Common7" / "Tools" / "VsDevCmd.bat",
            program_files / "Microsoft Visual Studio" / "2022" / "BuildTools" / "Common7" / "Tools" / "VsDevCmd.bat",
        ]
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


class EgoMotionCompensationContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def compile_and_run(self, source: str) -> str:
        vsdev = find_vsdev()
        if vsdev is None:
            self.skipTest("Visual Studio developer command prompt is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cpp = tmp_path / "ego_motion_contract.cpp"
            exe = tmp_path / "ego_motion_contract.exe"
            build_cmd = tmp_path / "build_ego_motion_contract.cmd"
            obj_dir = tmp_path / "obj"
            obj_dir.mkdir()
            cpp.write_text(source, encoding="utf-8")
            build_cmd.write_text(
                "\n".join(
                    [
                        "@echo off",
                        f'call "{vsdev}" -arch=x64 -host_arch=x64 >nul',
                        f'cl /nologo /std:c++17 /EHsc /I"{REPO_ROOT}" /I"{REPO_ROOT / "include"}" '
                        f'/Fo"{str(obj_dir)}\\\\" "{cpp}" /Fe:"{exe}"',
                    ]
                ),
                encoding="utf-8",
            )
            compile_result = subprocess.run(
                ["cmd", "/c", str(build_cmd)],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout)

            run_result = subprocess.run(
                [str(exe)],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stdout)
            return run_result.stdout

    def test_compensator_runtime_contract_exists(self):
        header_path = REPO_ROOT / "include" / "ego_motion_compensator.h"
        self.assertTrue(header_path.exists())
        header = self.read("include/ego_motion_compensator.h")

        for token in (
            "class EgoMotionCompensator",
            "struct EgoMotionSettings",
            "struct EgoMotionShift",
            "recordDelta",
            "consume",
            "reset",
            "std::deque",
            "std::mutex",
            "maxAgeMs",
            "maxShiftPx",
            "strength",
            "clampLength",
        ):
            self.assertIn(token, header)

        self.assertIn("applyPositionOffset", self.read("include/aim_kalman.h"))
        self.assertIn("applyPositionOffset", self.read("include/aim_imm.h"))

    def test_config_gui_and_runtime_wiring_are_default_off(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")
        loop = self.read("runtime/mouse_thread_loop.cpp")
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "ego_motion_compensation_enabled",
            "ego_motion_compensation_strength",
            "ego_motion_compensation_max_shift_px",
            "ego_motion_compensation_max_age_ms",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_mouse)

        self.assertIn("ego_motion_compensation_enabled = false", config_cpp)
        self.assertIn('get_bool("ego_motion_compensation_enabled", false)', config_cpp)
        self.assertIn('MERGE_FIELD("ego_motion_compensation_enabled"', config_cpp)
        self.assertIn('"ego_motion_compensation_enabled = "', config_cpp)
        self.assertIn("Ego-motion compensation", draw_mouse)

        for token in (
            "aim::EgoMotionCompensator",
            "egoMotionCompensator",
            "recordEgoMotionDelta",
            "consumeEgoMotionCompensation",
            "resetEgoMotionCompensation",
            "recordEgoMotionDelta(pixelDx, pixelDy, now)",
            "updateLastAppliedMouseDelta(pixelDx, pixelDy)",
            "egoMotionCompensator.recordDelta",
        ):
            self.assertIn(token, mouse_h + mouse_cpp)

        self.assertIn("consumeEgoMotionCompensation", loop)
        self.assertIn("egoMotionShift", loop)
        self.assertIn("targetTracker.updateAt(", loop)
        self.assertIn("trackerFrameTimestamp", loop)
        self.assertIn("const cv::Point2d& egoMotionShift", target_h + target_cpp)
        self.assertIn("compensatedEgoMotion", target_cpp)
        self.assertIn("applyInnerAimEgoMotion", target_cpp)
        self.assertIn("- compensatedEgoMotion.x", target_cpp)
        self.assertIn("newCx - compensatedOldCx", target_cpp)
        self.assertIn("egoMotionX", target_h + target_cpp)
        self.assertNotIn("command.pixelDx += ego", mouse_cpp)

    def test_ego_motion_compensator_accumulates_clamps_and_consumes_by_time_window(self):
        source = textwrap.dedent(
            r'''
            #include <chrono>
            #include <cmath>
            #include <iostream>

            #include "include/ego_motion_compensator.h"

            bool require(bool condition, const char* message)
            {
                if (!condition)
                {
                    std::cerr << message << "\n";
                    return false;
                }
                return true;
            }

            int main()
            {
                aim::EgoMotionSettings settings;
                settings.enabled = true;
                settings.strength = 0.5;
                settings.maxShiftPx = 8.0;
                settings.maxAgeMs = 120;

                aim::EgoMotionCompensator c;
                c.setSettings(settings);

                const auto t0 = std::chrono::steady_clock::now();
                c.recordDelta(10.0, 0.0, t0 + std::chrono::milliseconds(5));
                c.recordDelta(10.0, 0.0, t0 + std::chrono::milliseconds(10));
                c.recordDelta(1000.0, 0.0, t0 + std::chrono::milliseconds(200));

                const auto first = c.consume(t0, t0 + std::chrono::milliseconds(16), t0 + std::chrono::milliseconds(16));
                const auto second = c.consume(t0, t0 + std::chrono::milliseconds(16), t0 + std::chrono::milliseconds(16));
                const auto stale = c.consume(t0 + std::chrono::milliseconds(180), t0 + std::chrono::milliseconds(240), t0 + std::chrono::milliseconds(400));

                bool ok = true;
                ok &= require(first.valid, "first window should contain motion");
                ok &= require(std::abs(first.dx - 8.0) < 1e-6, "strength and max clamp should bound dx");
                ok &= require(std::abs(first.dy) < 1e-6, "dy should remain zero");
                ok &= require(!second.valid && std::abs(second.dx) < 1e-6, "consume should drain used samples");
                ok &= require(!stale.valid, "stale samples should fail closed");
                return ok ? 0 : 1;
            }
            '''
        )
        self.compile_and_run(source)


if __name__ == "__main__":
    unittest.main()
