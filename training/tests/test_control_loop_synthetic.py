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


class ControlLoopSyntheticTests(unittest.TestCase):
    def compile_and_run(self, source: str) -> str:
        vsdev = Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat")
        if not vsdev.exists():
            self.skipTest("Visual Studio developer command prompt is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cpp = tmp_path / "control_loop_synthetic.cpp"
            exe = tmp_path / "control_loop_synthetic.exe"
            build_cmd = tmp_path / "build_control_loop_synthetic.cmd"
            cpp.write_text(source, encoding="utf-8")

            build_cmd.write_text(
                "\n".join(
                    [
                        "@echo off",
                        f'call "{vsdev}" -arch=x64 -host_arch=x64 >nul',
                        f'cl /nologo /std:c++17 /EHsc '
                        f'/I"{REPO_ROOT}" /I"{REPO_ROOT / "mouse"}" /I"{REPO_ROOT / "include"}" '
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
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout)

            run_result = subprocess.run(
                [str(exe)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stdout)
            return run_result.stdout

    def test_state_estimator_pid_and_regression_metrics(self):
        source = textwrap.dedent(
            r'''
            #include <algorithm>
            #include <chrono>
            #include <cmath>
            #include <iostream>
            #include <vector>

            #include "include/aim_kalman.h"
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

            double syntheticTarget(double t)
            {
                double x = 40.0 * std::sin(2.0 * 3.14159265358979323846 * 0.75 * t);
                if (t >= 0.55)
                    x += 35.0;
                if (t >= 1.10)
                    x -= 70.0;
                return x;
            }

            constexpr double Pi = 3.14159265358979323846;

            double degToRad(double degrees)
            {
                return degrees * Pi / 180.0;
            }

            double radToDeg(double radians)
            {
                return radians * 180.0 / Pi;
            }

            double perspectiveAngleForOffset(double offsetPx, double resolutionPx, double fovDeg)
            {
                const double half = std::max(1.0, resolutionPx) * 0.5;
                const double normalized = offsetPx / half;
                const double halfFov = degToRad(std::clamp(fovDeg, 1.0, 179.0) * 0.5);
                return radToDeg(std::atan(std::tan(halfFov) * normalized));
            }

            double offsetForPerspectiveAngle(double angleDeg, double resolutionPx, double fovDeg)
            {
                const double half = std::max(1.0, resolutionPx) * 0.5;
                const double halfFov = degToRad(std::clamp(fovDeg, 1.0, 179.0) * 0.5);
                return half * std::tan(degToRad(angleDeg)) / std::max(1e-9, std::tan(halfFov));
            }

            double perspectiveCenterDegPerPx(double resolutionPx, double fovDeg)
            {
                const double half = std::max(1.0, resolutionPx) * 0.5;
                const double halfFov = degToRad(std::clamp(fovDeg, 1.0, 179.0) * 0.5);
                return radToDeg(std::tan(halfFov) / half);
            }

            aim::PidMouseSettings angularPidTestSettings(bool perspective)
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = true;
                settings.perspectiveFovMappingEnabled = perspective;
                settings.projectionWidthPx = 640.0;
                settings.projectionHeightPx = 640.0;
                settings.fovXDeg = 106.0;
                settings.fovYDeg = 74.0;
                settings.actuatorHz = 240;
                settings.kp = 0.060;
                settings.ki = 0.0;
                settings.kd = 0.0;
                settings.deadzonePx = 0.0;
                settings.maxPixelStep = 1000.0;
                settings.outputScale = 1.0;
                settings.minOutputScale = 1.0;
                settings.maxOutputScale = 1.0;
                settings.sizeReferencePx = 48.0;
                settings.sizeMinScale = 1.0;
                settings.sizeMaxScale = 1.0;
                settings.precisionRadiusScale = 0.0;
                settings.slowdownRadiusScale = 0.01;
                settings.feedForwardEnabled = false;
                settings.adaptiveOutputScalingEnabled = false;
                settings.conditionalIntegrationEnabled = false;
                return settings;
            }

            bool testPerspectiveFovPidMappingContract()
            {
                const double resolution = 640.0;
                const double fovX = 106.0;
                const double edgeOffset = 0.90 * resolution * 0.5;
                const double edgeAngle = perspectiveAngleForOffset(edgeOffset, resolution, fovX);
                const double centerDegPerPx = perspectiveCenterDegPerPx(resolution, fovX);
                const double expectedControlError = edgeAngle / centerDegPerPx;
                const double mirroredAngle = perspectiveAngleForOffset(-edgeOffset, resolution, fovX);
                const double linearAngle = edgeOffset * fovX / resolution;
                const double centerSlope = perspectiveAngleForOffset(1.0, resolution, fovX);
                const double edgeSlope = perspectiveAngleForOffset(edgeOffset, resolution, fovX) -
                    perspectiveAngleForOffset(edgeOffset - 1.0, resolution, fovX);

                aim::PidMouseController controller;
                controller.setSettings(angularPidTestSettings(true));

                aim::PidMouseObservation obs;
                obs.x = resolution * 0.5 + edgeOffset;
                obs.y = resolution * 0.5;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;
                obs.timestamp = Clock::now();

                controller.updateObservation(obs, resolution * 0.5, resolution * 0.5);
                auto cmd = controller.step(obs.timestamp);

                bool ok = true;
                ok &= require(std::abs(edgeAngle + mirroredAngle) < 1e-6, "perspective mapping should be symmetric around center");
                ok &= require(edgeSlope < centerSlope * 0.75, "perspective edge slope should compress relative to center");
                ok &= require(std::abs(edgeAngle - linearAngle) > 2.0, "edge perspective angle should differ from linear FOV angle");
                ok &= require(std::abs(cmd.errorX - expectedControlError) < 1e-3, "PID error should use perspective angular control units");
                ok &= require(cmd.angularOutputActive, "perspective PID should emit angular output");
                ok &= require(std::abs(cmd.angularDxDeg - cmd.pixelDx * centerDegPerPx) < 1e-6, "angular output should match center-scaled control output");
                return ok;
            }

            struct AngularStepMetrics
            {
                double rmse = 0.0;
                double finalError = 0.0;
                double settleTime = -1.0;
            };

            AngularStepMetrics simulateAngularStepResponse(bool perspective)
            {
                aim::PidMouseController controller;
                controller.setSettings(angularPidTestSettings(perspective));

                const double resolution = 640.0;
                const double fovX = 106.0;
                const double dt = 1.0 / 240.0;
                const double initialOffset = 0.90 * resolution * 0.5;
                double trueAngle = perspectiveAngleForOffset(initialOffset, resolution, fovX);
                const auto start = Clock::now();

                aim::PidMouseObservation obs;
                obs.y = resolution * 0.5;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;

                double squareError = 0.0;
                int samples = 0;
                AngularStepMetrics metrics;

                for (int i = 0; i < 300; ++i)
                {
                    const double offset = offsetForPerspectiveAngle(trueAngle, resolution, fovX);
                    obs.x = resolution * 0.5 + offset;
                    obs.timestamp = start + std::chrono::microseconds(static_cast<int>(i * dt * 1000000.0));
                    controller.updateObservation(obs, resolution * 0.5, resolution * 0.5);
                    const auto cmd = controller.step(obs.timestamp);

                    const double commandDeg = (perspective && cmd.angularOutputActive)
                        ? cmd.angularDxDeg
                        : cmd.pixelDx * fovX / resolution;
                    trueAngle -= commandDeg;

                    squareError += trueAngle * trueAngle;
                    ++samples;

                    if (metrics.settleTime < 0.0 && std::abs(trueAngle) < 0.75)
                        metrics.settleTime = i * dt;
                }

                metrics.rmse = std::sqrt(squareError / std::max(1, samples));
                metrics.finalError = std::abs(trueAngle);
                return metrics;
            }

            bool testPerspectiveFovPidEffectiveness()
            {
                const AngularStepMetrics linear = simulateAngularStepResponse(false);
                const AngularStepMetrics perspective = simulateAngularStepResponse(true);

                bool ok = true;
                ok &= require(perspective.finalError < 0.20, "perspective PID final angular error should settle near zero");
                ok &= require(perspective.rmse < linear.rmse * 0.92, "perspective PID should reduce edge-of-FOV angular RMSE vs linear mapping");
                ok &= require(perspective.settleTime >= 0.0 && perspective.settleTime < linear.settleTime, "perspective PID should settle faster at the edge of FOV");
                return ok;
            }

            bool testKalmanAcquisition()
            {
                aim::AimKalmanSettings settings;
                settings.enabled = true;
                settings.runtimeLatencySweepEnabled = true;
                settings.velocitySeedEnabled = true;
                settings.acquisitionFrames = 4;
                settings.process_noise_position = 120.0;
                settings.process_noise_velocity = 4500.0;
                settings.measurement_noise = 8.0;
                settings.velocity_damping = 0.04;

                aim::AimKalman2D kalman;
                kalman.setSettings(settings);
                auto t0 = kalman.update(0.0, 0.0, 1.0 / 120.0, 0.0);
                auto t1 = kalman.update(10.0, 0.0, 1.0 / 120.0, 2.0 / 120.0);

                bool ok = true;
                ok &= require(t0.prediction_weight <= 0.01, "first acquisition frame should not use prediction");
                ok &= require(t1.velocity_x > 250.0, "velocity should be seeded from early measurement deltas");
                ok &= require(t1.prediction_weight > 0.0 && t1.prediction_weight < 1.0, "prediction should ramp during acquisition");
                ok &= require(t1.predicted_x > t1.estimate_x, "ramped prediction should lead a moving target during acquisition");
                return ok;
            }

            bool testKalmanSyntheticMetrics()
            {
                aim::AimKalmanSettings settings;
                settings.enabled = true;
                settings.runtimeLatencySweepEnabled = true;
                settings.velocitySeedEnabled = true;
                settings.acquisitionFrames = 4;
                settings.process_noise_position = 140.0;
                settings.process_noise_velocity = 5200.0;
                settings.measurement_noise = 10.0;
                settings.velocity_damping = 0.05;

                aim::AimKalman2D kalman;
                kalman.setSettings(settings);

                const double dt = 1.0 / 120.0;
                const double lookahead = 2.0 / 120.0;
                double squareError = 0.0;
                int samples = 0;
                double settleAfterStep = -1.0;
                double settleAfterAbrupt = -1.0;

                for (int i = 0; i < 240; ++i)
                {
                    const double t = i * dt;
                    const double measurement = syntheticTarget(t) + 1.5 * std::sin(37.0 * t);
                    auto telemetry = kalman.update(measurement, 0.0, dt, lookahead);
                    const double truth = syntheticTarget(t + lookahead);
                    const double err = telemetry.predicted_x - truth;
                    if (i > 8)
                    {
                        squareError += err * err;
                        ++samples;
                    }
                    if (t >= 0.55 && settleAfterStep < 0.0 && std::abs(err) < 9.0)
                        settleAfterStep = t - 0.55;
                    if (t >= 1.10 && settleAfterAbrupt < 0.0 && std::abs(err) < 13.0)
                        settleAfterAbrupt = t - 1.10;
                }

                const double rmse = std::sqrt(squareError / std::max(1, samples));
                bool ok = true;
                ok &= require(rmse < 18.0, "synthetic sine/step/abrupt RMSE exceeded bound");
                ok &= require(settleAfterStep >= 0.0 && settleAfterStep < 1.0, "step settling time exceeded 1.0s");
                ok &= require(settleAfterAbrupt >= 0.0 && settleAfterAbrupt < 1.0, "abrupt settling time exceeded 1.0s");
                return ok;
            }

            bool testPidAntiWindupAndFeedForward()
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = true;
                settings.actuatorHz = 120;
                settings.kp = 0.010;
                settings.ki = 0.400;
                settings.kd = 0.0002;
                settings.maxPixelStep = 0.10;
                settings.outputScale = 1.0;
                settings.minOutputScale = 0.05;
                settings.maxOutputScale = 1.0;
                settings.maxIntegral = 10000.0;
                settings.conditionalIntegrationEnabled = true;
                settings.conditionalIntegrationErrorPx = 12.0;
                settings.adaptiveOutputScalingEnabled = true;
                settings.adaptiveOutputErrorScale = 60.0;
                settings.derivativeSmoothingMultiplier = 2.0;
                settings.feedForwardEnabled = true;
                settings.feedForwardGain = 3.0;
                settings.feedForwardFrameLookahead = 2;
                settings.feedForwardMaxStep = 5.0;
                settings.feedForwardMinSpeed = 1.0;
                settings.feedForwardConfidenceFloor = 0.0;

                aim::PidMouseController controller;
                controller.setSettings(settings);
                bool ok = true;
                ok &= require(controller.getSettings().feedForwardGain > 2.5, "feed-forward gain range should exceed 2.0");

                const auto start = Clock::now();
                aim::PidMouseObservation obs;
                obs.x = 260.0;
                obs.y = 160.0;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;

                for (int i = 0; i < 80; ++i)
                {
                    obs.timestamp = start + std::chrono::milliseconds(i * 8);
                    controller.updateObservation(obs, 160.0, 160.0);
                    auto cmd = controller.step(obs.timestamp);
                    if (i > 25)
                        ok &= require(std::abs(cmd.iX) < 0.05, "conditional integration should suppress windup while saturated");
                }

                controller.reset();
                obs.x = 110.0;
                obs.timestamp = start + std::chrono::milliseconds(700);
                controller.updateObservation(obs, 160.0, 160.0);
                auto cmd0 = controller.step(obs.timestamp);
                obs.x = 130.0;
                obs.timestamp = start + std::chrono::milliseconds(708);
                controller.updateObservation(obs, 160.0, 160.0);
                auto cmd1 = controller.step(obs.timestamp);

                ok &= require(cmd1.feedForwardActive, "feed-forward should activate from recent target velocity");
                ok &= require(std::abs(cmd1.feedForwardX) >= std::abs(cmd0.feedForwardX), "frame lookahead should not shrink feed-forward");
                return ok;
            }

            bool testPidSuppressesTangentialOrbitingNearTarget()
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = true;
                settings.actuatorHz = 120;
                settings.kp = 0.0085;
                settings.ki = 0.0003;
                settings.kd = 0.0001;
                settings.maxPixelStep = 20.0;
                settings.outputScale = 0.21;
                settings.minOutputScale = 0.02;
                settings.maxOutputScale = 0.35;
                settings.sizeReferencePx = 48.0;
                settings.sizeMinScale = 1.0;
                settings.sizeMaxScale = 1.0;
                settings.precisionRadiusScale = 0.012;
                settings.slowdownRadiusScale = 0.30;
                settings.feedForwardEnabled = true;
                settings.feedForwardGain = 4.0;
                settings.feedForwardLookaheadMs = 24.0;
                settings.feedForwardFrameLookahead = 2;
                settings.feedForwardMaxStep = 5.0;
                settings.feedForwardMinSpeed = 0.0;
                settings.feedForwardConfidenceFloor = 0.0;
                settings.conditionalIntegrationEnabled = true;
                settings.conditionalIntegrationErrorPx = 12.0;
                settings.adaptiveOutputScalingEnabled = true;
                settings.adaptiveOutputErrorScale = 96.0;

                aim::PidMouseController controller;
                controller.setSettings(settings);

                const auto start = Clock::now();
                aim::PidMouseObservation obs;
                obs.x = 165.0;
                obs.y = 150.0;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;
                obs.timestamp = start;
                controller.updateObservation(obs, 160.0, 160.0);

                obs.y = 160.0;
                obs.timestamp = start + std::chrono::microseconds(8333);
                controller.updateObservation(obs, 160.0, 160.0);
                auto cmd = controller.step(obs.timestamp);

                const double radial = cmd.pixelDx;
                const double tangential = cmd.pixelDy;
                bool ok = true;
                ok &= require(radial > 0.0, "near-target command should still move toward target");
                ok &= require(
                    std::abs(tangential) <= std::max(0.025, std::abs(radial) * 0.40),
                    "near-target tangential feed-forward should not dominate radial convergence");
                return ok;
            }

            bool testPidConvergesAtFullSpeedBetweenSparseFrames()
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
                settings.perspectiveFovMappingEnabled = false;

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

                bool ok = true;
                ok &= require(totalTravel > 105.0, "PID should keep moving at full speed between sparse detector frames");
                ok &= require(totalTravel < 125.0, "PID should not overshoot the observed target while detector cadence is sparse");
                return ok;
            }

            bool testDerivativeSmoothingAndDisabledSweepPerformance()
            {
                aim::PidMouseSettings settings;
                settings.runtimeLatencySweepEnabled = false;
                settings.actuatorHz = 1000;
                settings.derivativeSmoothingMultiplier = 4.0;
                settings.kp = 0.0085;
                settings.ki = 0.0003;
                settings.kd = 0.0001;

                aim::PidMouseController controller;
                controller.setSettings(settings);

                aim::PidMouseObservation obs;
                obs.x = 220.0;
                obs.y = 160.0;
                obs.width = 48.0;
                obs.height = 48.0;
                obs.confidence = 1.0;
                obs.valid = true;

                const auto start = Clock::now();
                const auto benchStart = Clock::now();
                double jitter = 0.0;
                double prev = 0.0;
                for (int i = 0; i < 2000; ++i)
                {
                    obs.x = 220.0 + ((i % 2 == 0) ? 0.75 : -0.75);
                    obs.timestamp = start + std::chrono::microseconds(i * 1000);
                    controller.updateObservation(obs, 160.0, 160.0);
                    auto cmd = controller.step(obs.timestamp);
                    if (i > 0)
                        jitter += std::abs(cmd.pixelDx - prev);
                    prev = cmd.pixelDx;
                }
                const auto benchEnd = Clock::now();
                const double avgStepMs = std::chrono::duration<double, std::milli>(benchEnd - benchStart).count() / 2000.0;

                bool ok = true;
                ok &= require(avgStepMs < 8.0, "PID step exceeded 8ms average with sweep off");
                ok &= require(jitter / 1999.0 < 0.80, "noise-injected actuator jitter exceeded threshold");
                return ok;
            }

            int main()
            {
                bool ok = true;
                ok &= testPerspectiveFovPidMappingContract();
                ok &= testPerspectiveFovPidEffectiveness();
                ok &= testKalmanAcquisition();
                ok &= testKalmanSyntheticMetrics();
                ok &= testPidAntiWindupAndFeedForward();
                ok &= testPidSuppressesTangentialOrbitingNearTarget();
                ok &= testPidConvergesAtFullSpeedBetweenSparseFrames();
                ok &= testDerivativeSmoothingAndDisabledSweepPerformance();
                if (ok)
                    std::cout << "control loop synthetic checks passed\n";
                return ok ? 0 : 1;
            }
            '''
        )
        output = self.compile_and_run(source)
        self.assertIn("control loop synthetic checks passed", output)


if __name__ == "__main__":
    unittest.main()
