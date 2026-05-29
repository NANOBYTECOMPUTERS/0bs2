import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "PidMouseController.cpp").exists()
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


class Phase4SmartBlendingContractTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (REPO_ROOT / relative).read_text(encoding="utf-8", errors="ignore")

    def test_smart_blender_module_is_project_owned(self):
        self.assertTrue((REPO_ROOT / "mouse" / "SmartBlender.h").exists())
        self.assertTrue((REPO_ROOT / "mouse" / "SmartBlender.cpp").exists())

        header = self.read("mouse/SmartBlender.h")
        source = self.read("mouse/SmartBlender.cpp")
        for token in [
            "struct SmartBlenderSettings",
            "struct SmartBlendInput",
            "struct SmartBlendOutput",
            "class SmartBlender",
            "setSettings",
            "apply",
            "reset",
        ]:
            self.assertIn(token, header)
        self.assertIn("fail closed", source.lower())

        for project in ["0BS_box_2.vcxproj", "cuda/0BS_cuda.vcxproj"]:
            xml = self.read(project)
            self.assertIn("mouse\\SmartBlender.cpp", xml)
            self.assertIn("mouse\\SmartBlender.h", xml)

    def test_config_defaults_gui_and_runtime_mapping_are_default_off(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for field in [
            "pid_smart_blending_enabled",
            "pid_smart_blending_aggression",
            "pid_smart_blending_near_damping",
            "pid_smart_blending_deadzone_px",
            "pid_smart_blending_jerk_limit_px",
            "pid_smart_blending_confidence_floor",
        ]:
            self.assertIn(field, config_h)
            self.assertIn(field, config_cpp)
            self.assertIn(field, draw_mouse)

        self.assertIn("pid_smart_blending_enabled = false", config_cpp)
        self.assertIn("pid_smart_blending_aggression = 0.65f", config_cpp)
        self.assertIn("pid_smart_blending_near_damping = 0.75f", config_cpp)

        self.assertIn("settings.pid_smart_blending_enabled = config.pid_smart_blending_enabled", mouse_cpp)
        self.assertIn("settings.pid_smart_blending_aggression", mouse_cpp)

    def test_pid_integration_shapes_output_after_feed_forward_only(self):
        controller_h = self.read("mouse/PidMouseController.h")
        controller_cpp = self.read("mouse/PidMouseController.cpp")

        self.assertIn('#include "SmartBlender.h"', controller_h)
        self.assertIn("SmartBlender smartBlender", controller_h)
        self.assertIn("pid_smart_blending_enabled", controller_h)
        self.assertIn("smartBlendActive", controller_h)
        self.assertIn("smartBlender.apply", controller_cpp)

        feedforward_pos = controller_cpp.index("outX += command.feedForwardX")
        blend_pos = controller_cpp.index("smartBlender.apply")
        self.assertGreater(blend_pos, feedforward_pos)
        self.assertIn("applyConvergenceDirectionGuard(outX, outY", controller_cpp[blend_pos:])

    def compile_and_run(self, source: str) -> str:
        vsdev = find_vsdev()
        if vsdev is None:
            self.skipTest("Visual Studio developer command prompt is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cpp = tmp_path / "phase4_smart_blending.cpp"
            exe = tmp_path / "phase4_smart_blending.exe"
            build_cmd = tmp_path / "build_phase4_smart_blending.cmd"
            obj_dir = tmp_path / "obj"
            obj_dir.mkdir()
            obj_dir_arg = str(obj_dir) + "\\\\"
            pdb = tmp_path / "phase4_smart_blending.pdb"
            cpp.write_text(source, encoding="utf-8")

            build_cmd.write_text(
                "\n".join(
                    [
                        "@echo off",
                        f'call "{vsdev}" -arch=x64 -host_arch=x64 >nul',
                        f'cl /nologo /std:c++17 /EHsc '
                        f'/I"{REPO_ROOT}" /I"{REPO_ROOT / "mouse"}" /I"{REPO_ROOT / "include"}" '
                        f'/Fo"{obj_dir_arg}" /Fd"{pdb}" '
                        f'"{cpp}" '
                        f'"{REPO_ROOT / "mouse" / "PidMouseController.cpp"}" '
                        f'"{REPO_ROOT / "mouse" / "PidGovernor.cpp"}" '
                        f'"{REPO_ROOT / "mouse" / "SmartBlender.cpp"}" '
                        f'/Fe:"{exe}"',
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

    def test_synthetic_jitter_and_full_speed_contract(self):
        source = textwrap.dedent(
            r'''
            #include <chrono>
            #include <cmath>
            #include <iostream>

            #include "mouse/PidMouseController.h"

            using Clock = std::chrono::steady_clock;

            bool require(bool condition, const char* message)
            {
                if (!condition)
                {
                    std::cerr << message << "\n";
                    return false;
                }
                return true;
            }

            struct Metrics
            {
                double jitter = 0.0;
                double jerk = 0.0;
                double travel = 0.0;
                double activeBlendFrames = 0.0;
            };

            aim::PidMouseSettings baseSettings()
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = true;
                settings.actuatorHz = 240;
                settings.kp = 0.090;
                settings.ki = 0.0;
                settings.kd = 0.0002;
                settings.deadzonePx = 0.0;
                settings.maxPixelStep = 8.0;
                settings.outputScale = 0.80;
                settings.minOutputScale = 0.20;
                settings.maxOutputScale = 1.0;
                settings.sizeReferencePx = 48.0;
                settings.sizeMinScale = 1.0;
                settings.sizeMaxScale = 1.0;
                settings.precisionRadiusScale = 0.0;
                settings.slowdownRadiusScale = 0.45;
                settings.feedForwardEnabled = false;
                settings.conditionalIntegrationEnabled = false;
                settings.adaptiveOutputScalingEnabled = false;
                settings.pid_smart_blending_aggression = 0.52;
                settings.pid_smart_blending_near_damping = 0.82;
                settings.pid_smart_blending_deadzone_px = 0.0;
                settings.pid_smart_blending_jerk_limit_px = 0.32;
                settings.pid_smart_blending_confidence_floor = 0.30;
                return settings;
            }

            Metrics runNearTargetJitter(bool smartBlend)
            {
                auto settings = baseSettings();
                settings.pid_smart_blending_enabled = smartBlend;

                aim::PidMouseController controller;
                controller.setSettings(settings);

                aim::PidMouseObservation obs;
                obs.y = 160.0;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 0.92;
                obs.valid = true;

                const auto start = Clock::now();
                double previous = 0.0;
                double previousDelta = 0.0;
                Metrics metrics;
                for (int i = 0; i < 220; ++i)
                {
                    const double alternating = (i % 2 == 0) ? 5.0 : -5.0;
                    obs.x = 182.0 + alternating + 2.0 * std::sin(i * 0.31);
                    obs.timestamp = start + std::chrono::microseconds(i * 4167);
                    controller.updateObservation(obs, 160.0, 160.0);
                    const auto cmd = controller.step(obs.timestamp);
                    const double delta = cmd.pixelDx - previous;
                    if (i > 15)
                    {
                        metrics.jitter += std::abs(delta);
                        metrics.jerk += std::abs(delta - previousDelta);
                    }
                    metrics.travel += std::hypot(cmd.pixelDx, cmd.pixelDy);
                    metrics.activeBlendFrames += cmd.smartBlendActive ? 1.0 : 0.0;
                    previous = cmd.pixelDx;
                    previousDelta = delta;
                }
                return metrics;
            }

            double runSparseFrameTravel()
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = false;
                settings.actuatorHz = 1000;
                settings.kp = 0.20;
                settings.ki = 0.0;
                settings.kd = 0.0;
                settings.deadzonePx = 0.0;
                settings.maxPixelStep = 20.0;
                settings.outputScale = 3.0;
                settings.minOutputScale = 3.0;
                settings.maxOutputScale = 3.0;
                settings.sizeReferencePx = 48.0;
                settings.sizeMinScale = 1.0;
                settings.sizeMaxScale = 1.0;
                settings.precisionRadiusScale = 0.012;
                settings.slowdownRadiusScale = 0.30;
                settings.feedForwardEnabled = false;
                settings.conditionalIntegrationEnabled = false;
                settings.adaptiveOutputScalingEnabled = false;
                settings.pid_smart_blending_enabled = false;

                aim::PidMouseController controller;
                controller.setSettings(settings);

                const auto start = Clock::now();
                aim::PidMouseObservation obs;
                obs.x = 280.0;
                obs.y = 160.0;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;
                obs.timestamp = start;
                controller.updateObservation(obs, 160.0, 160.0);

                double totalTravel = 0.0;
                for (int i = 0; i < 50; ++i)
                {
                    const auto tick = start + std::chrono::microseconds(i * 1000);
                    const auto cmd = controller.step(tick);
                    totalTravel += std::hypot(cmd.pixelDx, cmd.pixelDy);
                }
                return totalTravel;
            }

            int main()
            {
                const Metrics raw = runNearTargetJitter(false);
                const Metrics blended = runNearTargetJitter(true);
                const double sparseTravel = runSparseFrameTravel();

                bool ok = true;
                ok &= require(blended.activeBlendFrames > 150.0, "smart blend should be active when enabled");
                ok &= require(blended.jitter < raw.jitter * 0.78, "smart blend should reduce near-target output jitter");
                ok &= require(blended.jerk < raw.jerk * 0.72, "smart blend should reduce actuator jerk");
                ok &= require(sparseTravel > 105.0 && sparseTravel < 125.0, "default-off path should preserve full-speed sparse-frame travel");
                if (ok)
                    std::cout << "phase4 smart blending synthetic checks passed\n";
                return ok ? 0 : 1;
            }
            '''
        )
        output = self.compile_and_run(source)
        self.assertIn("phase4 smart blending synthetic checks passed", output)


if __name__ == "__main__":
    unittest.main()
