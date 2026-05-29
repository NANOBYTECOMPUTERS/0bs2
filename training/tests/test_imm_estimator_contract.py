import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "include" / "aim_kalman.h").exists()
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


class ImmEstimatorContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def compile_and_run(self, source: str) -> str:
        vsdev = find_vsdev()
        if vsdev is None:
            self.skipTest("Visual Studio developer command prompt is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cpp = tmp_path / "imm_estimator_synthetic.cpp"
            exe = tmp_path / "imm_estimator_synthetic.exe"
            build_cmd = tmp_path / "build_imm_estimator_synthetic.cmd"
            obj_dir = tmp_path / "obj"
            obj_dir.mkdir()
            pdb = tmp_path / "imm_estimator_synthetic.pdb"
            cpp.write_text(source, encoding="utf-8")
            build_cmd.write_text(
                "\n".join(
                    [
                        "@echo off",
                        f'call "{vsdev}" -arch=x64 -host_arch=x64 >nul',
                        f'cl /nologo /std:c++17 /EHsc /I"{REPO_ROOT}" /I"{REPO_ROOT / "include"}" '
                        f'/Fo"{str(obj_dir)}\\\\" /Fd"{pdb}" "{cpp}" /Fe:"{exe}"',
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

    def test_imm_runtime_contract_tokens_exist(self):
        imm_h = REPO_ROOT / "include" / "aim_imm.h"
        self.assertTrue(imm_h.exists())
        imm = self.read("include/aim_imm.h")

        for token in (
            "class AimIMM2D",
            "AimKalmanSettings",
            "AimKalmanTelemetry",
            "struct ModeProbability",
            "modeProbabilities",
            "transition_probability",
            "constant_velocity",
            "constant_acceleration",
            "measurementLikelihood",
            "predictAxisAhead",
        ):
            self.assertIn(token, imm)

    def test_config_gui_and_tracker_wire_imm_mode_without_deleting_legacy_knobs(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "std::string estimator_mode",
            'estimator_mode = "kalman"',
            'get_string("estimator_mode", "kalman")',
            'MERGE_FIELD("estimator_mode", estimator_mode)',
            '"estimator_mode = "',
        ):
            self.assertIn(token, config_h + config_cpp)

        for legacy_token in (
            "kalman_additional_prediction_ms",
            "predictionInterval",
            "kalman_warmup_frames",
            "kalman_acquisition_frames",
            "kalman_velocity_seed_enabled",
        ):
            self.assertIn(legacy_token, config_h + config_cpp + draw_mouse)

        self.assertIn("Estimator mode", draw_mouse)
        self.assertIn("aim::AimIMM2D imm", target_h)
        self.assertIn('config.estimator_mode == "imm"', target_cpp)
        self.assertIn("track.imm.update", target_cpp)
        self.assertIn("t.innerAim.imm.predict", target_cpp)
        self.assertIn("aim::AimKalman2D", self.read("mouse/mouse.h"))
        self.assertIn("targetKalman", self.read("mouse/mouse.h"))
        self.assertNotIn("command.pixelDx += imm", self.read("mouse/PidMouseController.cpp"))

    def test_imm_compiles_and_switches_mode_probability_on_reversal(self):
        source = textwrap.dedent(
            r'''
            #include <algorithm>
            #include <cmath>
            #include <iostream>

            #include "include/aim_kalman.h"
            #include "include/aim_imm.h"

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
                aim::AimKalmanSettings settings;
                settings.enabled = true;
                settings.process_noise_position = 6.0;
                settings.process_noise_velocity = 1800.0;
                settings.measurement_noise = 2.0;
                settings.max_velocity = 60000.0;

                aim::AimIMM2D imm;
                imm.setSettings(settings);

                constexpr double dt = 1.0 / 120.0;
                double caBefore = 0.0;
                double caAfter = 0.0;
                double maxJump = 0.0;
                double previousEstimate = 0.0;

                for (int i = 0; i < 180; ++i)
                {
                    const double t = i * dt;
                    double truth = 0.0;
                    if (t < 0.55)
                        truth = 480.0 * t;
                    else
                        truth = 480.0 * 0.55 - 620.0 * (t - 0.55);
                    const double measurement = truth + std::sin(53.0 * t) * 0.65;

                    const auto telemetry = imm.update(measurement, 0.0, dt, 0.016);
                    const auto modes = imm.modeProbabilities();
                    if (t > 0.35 && t < 0.50)
                        caBefore = std::max(caBefore, modes.constant_acceleration);
                    if (t > 0.62 && t < 0.85)
                        caAfter = std::max(caAfter, modes.constant_acceleration);

                    if (i > 0)
                        maxJump = std::max(maxJump, std::abs(telemetry.estimate_x - previousEstimate));
                    previousEstimate = telemetry.estimate_x;
                }

                bool ok = true;
                ok &= require(imm.initialized(), "IMM should initialize from measurements");
                ok &= require(caAfter > caBefore + 0.08, "CA mode probability should increase after abrupt reversal");
                ok &= require(caAfter > 0.28, "CA mode probability should become material during maneuver");
                ok &= require(maxJump < 30.0, "IMM estimate should remain bounded without large jumps");
                return ok ? 0 : 1;
            }
            '''
        )
        self.compile_and_run(source)


if __name__ == "__main__":
    unittest.main()
