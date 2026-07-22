#include "BoxTarget.h"
#include "config.h"
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

Config config;

namespace
{
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 720;
constexpr double kPi = 3.14159265358979323846;
constexpr double kAverageAimErrorBudgetPx = 32.0;
constexpr double kMaxAimErrorBudgetPx = 46.0;

struct SyntheticBox
{
    cv::Rect box;
    cv::Point2d truthAim;
    float confidence = 1.0f;
    bool visible = false;
};

struct UnrealCameraState
{
    double yawRadians = 0.0;
    double pitchPixels = 0.0;
    double lateralOffset = 0.0;
};

struct UnrealActorState
{
    double lateralMeters = 0.0;
    double depthMeters = 10.0;
    double verticalPixels = 0.0;
    double heightMeters = 1.8;
};

struct ScenarioMetrics
{
    int lockSwitches = 0;
    int missingLockedFrames = 0;
    int visibleFrames = 0;
    int matchedLockedFrames = 0;
    int maxMissedFrames = 0;
    double sumAimErrorPx = 0.0;
    double maxAimErrorPx = 0.0;
};

void configureSyntheticTracker()
{
    config.class_player = 0;
    config.class_head = 1;
    config.disable_headshot = true;
    config.body_y_offset = Config::kBodyYOffsetDefault;
    config.head_y_offset = Config::kHeadYOffsetDefault;
    config.detection_resolution = 640;
    config.runtime_latency_sweep_enabled = Config::kRuntimeLatencySweepEnabledDefault;
    config.estimator_mode = "kalman";
    config.kalman_enabled = true;
    config.kalman_process_noise_position = 40.0f;
    config.kalman_process_noise_velocity = Config::kKalmanProcessNoiseVelocityDefault;
    config.kalman_measurement_noise = Config::kKalmanMeasurementNoiseDefault;
    config.kalman_velocity_damping = Config::kKalmanVelocityDampingDefault;
    config.kalman_max_velocity = 20000.0f;
    config.kalman_warmup_frames = 2;
    config.kalman_velocity_seed_enabled = true;
    config.kalman_acquisition_frames = Config::kKalmanAcquisitionFramesDefault;
    config.ego_motion_compensation_enabled = true;
    config.ego_motion_compensation_max_shift_px = 32.0f;
    config.tracker_v2_enabled = true;
    config.tracker_v2_high_confidence = 0.45f;
    config.tracker_v2_new_track_confidence = 0.55f;
    config.tracker_v2_detector_max_candidates = 160;
    config.tracker_v2_box_smoothing_alpha = 0.34f;
    config.tracker_v2_box_prediction_alpha = 0.18f;
}

double deterministicNoise(int frame, double phase, double amplitude)
{
    return amplitude * (0.62 * std::sin(static_cast<double>(frame) * 0.71 + phase) +
        0.38 * std::sin(static_cast<double>(frame) * 1.83 + phase * 0.37));
}

UnrealCameraState makeCamera(int frame)
{
    const double t = static_cast<double>(frame) / 120.0;
    UnrealCameraState camera;
    camera.yawRadians = 0.080 * std::sin(t * 2.4) + 0.020 * std::sin(t * 15.0);
    camera.pitchPixels = 16.0 * std::sin(t * 3.1);
    camera.lateralOffset = 0.22 * std::sin(t * 1.7);
    return camera;
}

UnrealActorState makePrimaryActor(int frame)
{
    const double t = static_cast<double>(frame) / 120.0;
    UnrealActorState actor;
    actor.lateralMeters = -1.0 + 2.6 * t + 0.18 * std::sin(t * 6.0);
    actor.depthMeters = 10.5 + 1.6 * std::sin(t * 2.0 + 0.25);
    actor.verticalPixels = 11.0 * std::sin(t * 5.0);
    return actor;
}

UnrealActorState makeCrossingActor(int frame)
{
    const double t = static_cast<double>(frame - 72) / 58.0;
    UnrealActorState actor;
    actor.lateralMeters = 1.25 - 2.35 * std::clamp(t, 0.0, 1.0);
    actor.depthMeters = 10.1 + 0.7 * std::sin(static_cast<double>(frame) * 0.09);
    actor.verticalPixels = -4.0 + 8.0 * std::sin(static_cast<double>(frame) * 0.04);
    actor.heightMeters = 1.74;
    return actor;
}

SyntheticBox projectActor(
    const UnrealActorState& actor,
    const UnrealCameraState& camera,
    int frame,
    float confidence,
    double jitterPhase)
{
    constexpr double horizontalFovRadians = 90.0 * kPi / 180.0;
    const double focalPixels = static_cast<double>(kScreenWidth) /
        (2.0 * std::tan(horizontalFovRadians * 0.5));

    const double angleX = std::atan2(actor.lateralMeters - camera.lateralOffset, actor.depthMeters) - camera.yawRadians;
    if (std::abs(angleX) > horizontalFovRadians * 0.48 || actor.depthMeters <= 0.5)
        return {};

    const double centerX = static_cast<double>(kScreenWidth) * 0.5 + std::tan(angleX) * focalPixels;
    const double centerY = static_cast<double>(kScreenHeight) * 0.5 + camera.pitchPixels + actor.verticalPixels;
    const double heightPx = std::clamp(actor.heightMeters / actor.depthMeters * focalPixels, 58.0, 190.0);
    const double widthPx = heightPx * 0.42;

    const double jitterX = deterministicNoise(frame, jitterPhase, 2.2);
    const double jitterY = deterministicNoise(frame, jitterPhase + 1.7, 1.7);
    const double jitterW = deterministicNoise(frame, jitterPhase + 3.1, 1.0);
    const double jitterH = deterministicNoise(frame, jitterPhase + 4.3, 2.0);

    const int x = static_cast<int>(std::lround(centerX - widthPx * 0.5 + jitterX));
    const int y = static_cast<int>(std::lround(centerY - heightPx * 0.5 + jitterY));
    const int w = std::max(8, static_cast<int>(std::lround(widthPx + jitterW)));
    const int h = std::max(16, static_cast<int>(std::lround(heightPx + jitterH)));

    if (x + w <= 0 || x >= kScreenWidth || y + h <= 0 || y >= kScreenHeight)
        return {};

    SyntheticBox out;
    out.box = cv::Rect(x, y, w, h);
    out.truthAim = {
        centerX,
        centerY - heightPx * 0.5 + heightPx * Config::kBodyYOffsetDefault
    };
    out.confidence = confidence;
    out.visible = true;
    return out;
}

ScenarioMetrics runUnrealStyleScenario()
{
    configureSyntheticTracker();
    MultiTargetTracker tracker;
    ScenarioMetrics metrics;
    int lockedTrackId = -1;

    for (int frame = 0; frame < 150; ++frame)
    {
        const UnrealCameraState camera = makeCamera(frame);
        const bool primaryOccluded = frame >= 58 && frame <= 64;
        const bool crossingActive = frame >= 72 && frame <= 130;

        std::vector<cv::Rect> boxes;
        std::vector<int> classes;
        std::vector<float> confidences;
        SyntheticBox primary;

        if (!primaryOccluded)
        {
            const float confidence = static_cast<float>(std::clamp(
                0.86 + 0.05 * std::sin(static_cast<double>(frame) * 0.07) -
                    ((frame > 105 && frame < 118) ? 0.18 : 0.0),
                0.58,
                0.94));
            primary = projectActor(makePrimaryActor(frame), camera, frame, confidence, 0.3);
            if (primary.visible)
            {
                boxes.push_back(primary.box);
                classes.push_back(config.class_player);
                confidences.push_back(primary.confidence);
                ++metrics.visibleFrames;
            }
        }

        if (crossingActive)
        {
            const float confidence = static_cast<float>(0.69 + 0.08 * std::sin(static_cast<double>(frame) * 0.11));
            const SyntheticBox decoy = projectActor(makeCrossingActor(frame), camera, frame, confidence, 2.4);
            if (decoy.visible)
            {
                boxes.push_back(decoy.box);
                classes.push_back(config.class_player);
                confidences.push_back(decoy.confidence);
            }
        }

        const cv::Point2d egoShift(
            deterministicNoise(frame, 4.8, 1.8),
            deterministicNoise(frame, 6.2, 1.2));
        tracker.update(
            boxes,
            classes,
            confidences,
            kScreenWidth,
            kScreenHeight,
            true,
            frame > 0,
            egoShift);

        LockedTargetInfo locked;
        if (!tracker.getLockedTarget(locked))
        {
            ++metrics.missingLockedFrames;
            continue;
        }

        if (lockedTrackId == -1)
            lockedTrackId = locked.trackId;
        else if (locked.trackId != lockedTrackId)
            ++metrics.lockSwitches;

        metrics.maxMissedFrames = std::max(metrics.maxMissedFrames, locked.missedFrames);

        if (primary.visible && locked.trackId == lockedTrackId)
        {
            const double error = std::hypot(
                locked.target.smoothX - primary.truthAim.x,
                locked.target.smoothY - primary.truthAim.y);
            metrics.sumAimErrorPx += error;
            metrics.maxAimErrorPx = std::max(metrics.maxAimErrorPx, error);
            ++metrics.matchedLockedFrames;
        }
    }

    return metrics;
}

void testUnrealStyleScenarioMaintainsTargetLock()
{
    const ScenarioMetrics metrics = runUnrealStyleScenario();
    std::cout << "Synthetic lock metrics: visible=" << metrics.visibleFrames
        << " matched=" << metrics.matchedLockedFrames
        << " missing=" << metrics.missingLockedFrames
        << " switches=" << metrics.lockSwitches
        << " max_missed=" << metrics.maxMissedFrames << "\n";
    REQUIRE(metrics.visibleFrames >= 130);
    REQUIRE(metrics.matchedLockedFrames >= 120);
    REQUIRE(metrics.missingLockedFrames <= 2);
    REQUIRE(metrics.lockSwitches == 0);
    REQUIRE(metrics.maxMissedFrames <= 7);
}

void testUnrealStyleScenarioKeepsAimErrorBounded()
{
    const ScenarioMetrics metrics = runUnrealStyleScenario();
    const double averageAimError = metrics.sumAimErrorPx /
        static_cast<double>(std::max(1, metrics.matchedLockedFrames));
    std::cout << "Synthetic aim metrics: avg_error=" << averageAimError
        << " max_error=" << metrics.maxAimErrorPx
        << " matched=" << metrics.matchedLockedFrames << "\n";

    REQUIRE(averageAimError < kAverageAimErrorBudgetPx);
    REQUIRE(metrics.maxAimErrorPx < kMaxAimErrorBudgetPx);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "Unreal-style scenario maintains target lock", testUnrealStyleScenarioMaintainsTargetLock },
            { "Unreal-style scenario keeps aim error bounded", testUnrealStyleScenarioKeepsAimErrorBounded },
        },
        "Unreal-style synthetic targeting");
}
