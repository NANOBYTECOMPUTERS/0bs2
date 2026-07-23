#include "BoxTarget.h"
#include "config.h"
#include "test_harness.h"

#include <algorithm>
#include <chrono>
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
constexpr float kSyntheticClosedLoopSharpness = 56.0f;
constexpr double kCenterConvergenceAverageBudgetPx = 9.0;
constexpr double kCenterConvergenceFinalBudgetPx = 7.0;
constexpr double kCenterConvergenceMaxBudgetPx = 22.0;

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

struct CenterConvergenceMetrics
{
    int visibleFrames = 0;
    int lockedFrames = 0;
    int lockSwitches = 0;
    int framesInsideEightPx = 0;
    int firstInsideEightFrame = -1;
    double initialErrorPx = 0.0;
    double finalErrorPx = 0.0;
    double averageErrorAfterWarmupPx = 0.0;
    double maxErrorAfterWarmupPx = 0.0;
    double totalPhysicalMovementPx = 0.0;
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
    config.target_deadzone_px = 0.0f;
    config.target_stream_sharpness = Config::kTargetStreamSharpnessDefault;
    config.target_max_pixel_speed = 1800.0f;
    config.target_min_stream_confidence = Config::kTargetMinStreamConfidenceDefault;
    config.target_prediction_blend = 0.18f;
    config.target_prediction_max_lead_px = 8.0f;
    config.target_calibrated_pixel_counts_enabled = true;
    config.target_counts_per_pixel_x = 1.0f;
    config.target_counts_per_pixel_y = 1.0f;
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

std::chrono::steady_clock::time_point syntheticFrameTimestamp(int frame)
{
    using clock = std::chrono::steady_clock;
    const auto elapsed = std::chrono::duration<double>(static_cast<double>(frame) / 120.0);
    return clock::time_point{} + std::chrono::duration_cast<clock::duration>(elapsed);
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

SyntheticBox applyVirtualCameraOffset(const SyntheticBox& box, double offsetX, double offsetY)
{
    if (!box.visible)
        return {};

    SyntheticBox out = box;
    out.box.x = static_cast<int>(std::lround(static_cast<double>(box.box.x) - offsetX));
    out.box.y = static_cast<int>(std::lround(static_cast<double>(box.box.y) - offsetY));
    out.truthAim.x -= offsetX;
    out.truthAim.y -= offsetY;

    if (out.box.x + out.box.width <= 0 || out.box.x >= kScreenWidth ||
        out.box.y + out.box.height <= 0 || out.box.y >= kScreenHeight)
    {
        return {};
    }

    return out;
}

std::pair<double, double> predictSyntheticStreamAim(
    const LockedTargetInfo& locked,
    double ageSec)
{
    ageSec = std::clamp(ageSec, 0.0, 0.250);
    double predictedX = locked.target.smoothX + locked.targetVelocityX * ageSec;
    double predictedY = locked.target.smoothY + locked.targetVelocityY * ageSec;

    if (!std::isfinite(predictedX) || !std::isfinite(predictedY))
        return { locked.target.smoothX, locked.target.smoothY };

    double leadX = predictedX - locked.target.smoothX;
    double leadY = predictedY - locked.target.smoothY;
    const double maxLead = std::clamp(static_cast<double>(config.target_prediction_max_lead_px), 0.0, 40.0);
    if (maxLead <= 1e-6)
        return { locked.target.smoothX, locked.target.smoothY };

    const double leadMagnitude = std::hypot(leadX, leadY);
    if (leadMagnitude > maxLead && leadMagnitude > 1e-9)
    {
        const double scale = maxLead / leadMagnitude;
        leadX *= scale;
        leadY *= scale;
    }

    const double minConfidence = std::clamp(static_cast<double>(config.target_min_stream_confidence), 0.0, 0.95);
    const double confidenceRange = std::max(0.05, 1.0 - minConfidence);
    const double confidenceGate = std::clamp(
        (locked.target.confidence - minConfidence) / confidenceRange,
        0.0,
        1.0);
    const double blend = std::clamp(static_cast<double>(config.target_prediction_blend), 0.0, 0.65);

    return {
        locked.target.smoothX + leadX * blend * confidenceGate,
        locked.target.smoothY + leadY * blend * confidenceGate
    };
}

double applySyntheticStreamTicks(
    const LockedTargetInfo& locked,
    double& cameraOffsetX,
    double& cameraOffsetY,
    double& countCarryX,
    double& countCarryY)
{
    constexpr int streamTicksPerFrame = 8;
    constexpr double streamDtSec = 0.001;
    double appliedSinceObservationX = 0.0;
    double appliedSinceObservationY = 0.0;
    double physicalMovementPx = 0.0;
    const double countsPerPixelX = std::abs(static_cast<double>(config.target_counts_per_pixel_x));
    const double countsPerPixelY = std::abs(static_cast<double>(config.target_counts_per_pixel_y));

    for (int tick = 0; tick < streamTicksPerFrame; ++tick)
    {
        const auto aim = predictSyntheticStreamAim(
            locked,
            static_cast<double>(tick + 1) * streamDtSec);
        double pixelDx = aim.first - static_cast<double>(kScreenWidth) * 0.5 - appliedSinceObservationX;
        double pixelDy = aim.second - static_cast<double>(kScreenHeight) * 0.5 - appliedSinceObservationY;
        if (!std::isfinite(pixelDx) || !std::isfinite(pixelDy))
            continue;

        const double distance = std::hypot(pixelDx, pixelDy);
        const double deadzone = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
        if (distance <= deadzone)
            continue;

        const double sharpness = std::clamp(static_cast<double>(config.target_stream_sharpness), 1.0, 80.0);
        const double alpha = std::clamp(1.0 - std::exp(-sharpness * streamDtSec), 0.0, 1.0);
        pixelDx *= alpha;
        pixelDy *= alpha;

        const double maxSpeed = std::clamp(static_cast<double>(config.target_max_pixel_speed), 50.0, 20000.0);
        const double maxStep = std::max(0.01, maxSpeed * streamDtSec);
        const double stepLength = std::hypot(pixelDx, pixelDy);
        if (stepLength > maxStep && stepLength > 1e-9)
        {
            const double scale = maxStep / stepLength;
            pixelDx *= scale;
            pixelDy *= scale;
        }

        appliedSinceObservationX += pixelDx;
        appliedSinceObservationY += pixelDy;

        countCarryX += pixelDx * countsPerPixelX;
        countCarryY += pixelDy * countsPerPixelY;
        const int emittedCountsX = static_cast<int>(std::round(countCarryX));
        const int emittedCountsY = static_cast<int>(std::round(countCarryY));
        countCarryX -= static_cast<double>(emittedCountsX);
        countCarryY -= static_cast<double>(emittedCountsY);

        const double physicalDx = countsPerPixelX > 1e-9 ? emittedCountsX / countsPerPixelX : emittedCountsX;
        const double physicalDy = countsPerPixelY > 1e-9 ? emittedCountsY / countsPerPixelY : emittedCountsY;
        cameraOffsetX += physicalDx;
        cameraOffsetY += physicalDy;
        physicalMovementPx += std::hypot(physicalDx, physicalDy);
    }

    return physicalMovementPx;
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
        tracker.updateAt(
            boxes,
            classes,
            confidences,
            kScreenWidth,
            kScreenHeight,
            true,
            frame > 0,
            syntheticFrameTimestamp(frame),
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

CenterConvergenceMetrics runUnrealClosedLoopConvergenceScenario()
{
    configureSyntheticTracker();
    config.target_stream_sharpness = kSyntheticClosedLoopSharpness;
    MultiTargetTracker tracker;
    CenterConvergenceMetrics metrics;
    int lockedTrackId = -1;
    double cameraOffsetX = 0.0;
    double cameraOffsetY = 0.0;
    double lastObservationCameraOffsetX = 0.0;
    double lastObservationCameraOffsetY = 0.0;
    double countCarryX = 0.0;
    double countCarryY = 0.0;
    double sumErrorAfterWarmup = 0.0;
    int warmupSamples = 0;

    for (int frame = 0; frame < 170; ++frame)
    {
        const UnrealCameraState camera = makeCamera(frame);
        const bool primaryOccluded = frame >= 58 && frame <= 64;
        const bool crossingActive = frame >= 72 && frame <= 130;

        std::vector<cv::Rect> boxes;
        std::vector<int> classes;
        std::vector<float> confidences;
        SyntheticBox primaryWorld;

        if (!primaryOccluded)
        {
            const float confidence = static_cast<float>(std::clamp(
                0.88 + 0.04 * std::sin(static_cast<double>(frame) * 0.07) -
                    ((frame > 105 && frame < 118) ? 0.16 : 0.0),
                0.60,
                0.95));
            primaryWorld = projectActor(makePrimaryActor(frame), camera, frame, confidence, 0.3);
            const SyntheticBox primaryObserved = applyVirtualCameraOffset(primaryWorld, cameraOffsetX, cameraOffsetY);
            if (primaryObserved.visible)
            {
                boxes.push_back(primaryObserved.box);
                classes.push_back(config.class_player);
                confidences.push_back(primaryObserved.confidence);
                ++metrics.visibleFrames;
            }
        }

        if (crossingActive)
        {
            const float confidence = static_cast<float>(0.67 + 0.08 * std::sin(static_cast<double>(frame) * 0.11));
            const SyntheticBox decoyWorld = projectActor(makeCrossingActor(frame), camera, frame, confidence, 2.4);
            const SyntheticBox decoyObserved = applyVirtualCameraOffset(decoyWorld, cameraOffsetX, cameraOffsetY);
            if (decoyObserved.visible)
            {
                boxes.push_back(decoyObserved.box);
                classes.push_back(config.class_player);
                confidences.push_back(decoyObserved.confidence);
            }
        }

        const cv::Point2d egoShift(
            cameraOffsetX - lastObservationCameraOffsetX,
            cameraOffsetY - lastObservationCameraOffsetY);
        tracker.updateAt(
            boxes,
            classes,
            confidences,
            kScreenWidth,
            kScreenHeight,
            true,
            frame > 0,
            syntheticFrameTimestamp(frame),
            egoShift);
        lastObservationCameraOffsetX = cameraOffsetX;
        lastObservationCameraOffsetY = cameraOffsetY;

        LockedTargetInfo locked;
        if (!tracker.getLockedTarget(locked))
            continue;

        if (lockedTrackId == -1)
            lockedTrackId = locked.trackId;
        else if (locked.trackId != lockedTrackId)
            ++metrics.lockSwitches;

        if (!primaryWorld.visible || locked.trackId != lockedTrackId)
            continue;

        ++metrics.lockedFrames;
        const double preError = std::hypot(
            primaryWorld.truthAim.x - cameraOffsetX - static_cast<double>(kScreenWidth) * 0.5,
            primaryWorld.truthAim.y - cameraOffsetY - static_cast<double>(kScreenHeight) * 0.5);
        if (metrics.lockedFrames == 1)
            metrics.initialErrorPx = preError;

        metrics.totalPhysicalMovementPx += applySyntheticStreamTicks(
            locked,
            cameraOffsetX,
            cameraOffsetY,
            countCarryX,
            countCarryY);

        const double postError = std::hypot(
            primaryWorld.truthAim.x - cameraOffsetX - static_cast<double>(kScreenWidth) * 0.5,
            primaryWorld.truthAim.y - cameraOffsetY - static_cast<double>(kScreenHeight) * 0.5);
        metrics.finalErrorPx = postError;
        if (postError <= 8.0)
        {
            ++metrics.framesInsideEightPx;
            if (metrics.firstInsideEightFrame < 0)
                metrics.firstInsideEightFrame = frame;
        }

        if (frame >= 34)
        {
            sumErrorAfterWarmup += postError;
            metrics.maxErrorAfterWarmupPx = std::max(metrics.maxErrorAfterWarmupPx, postError);
            ++warmupSamples;
        }
    }

    metrics.averageErrorAfterWarmupPx = sumErrorAfterWarmup /
        static_cast<double>(std::max(1, warmupSamples));
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

void testUnrealStyleClosedLoopConvergesToCenter()
{
    const CenterConvergenceMetrics metrics = runUnrealClosedLoopConvergenceScenario();
    std::cout << "Synthetic center convergence metrics: initial_error=" << metrics.initialErrorPx
        << " final_error=" << metrics.finalErrorPx
        << " avg_after_warmup=" << metrics.averageErrorAfterWarmupPx
        << " max_after_warmup=" << metrics.maxErrorAfterWarmupPx
        << " visible=" << metrics.visibleFrames
        << " locked=" << metrics.lockedFrames
        << " inside_8px=" << metrics.framesInsideEightPx
        << " first_inside_8_frame=" << metrics.firstInsideEightFrame
        << " movement_px=" << metrics.totalPhysicalMovementPx
        << " switches=" << metrics.lockSwitches << "\n";

    REQUIRE(metrics.visibleFrames >= 145);
    REQUIRE(metrics.lockedFrames >= 140);
    REQUIRE(metrics.lockSwitches == 0);
    REQUIRE(metrics.firstInsideEightFrame >= 0);
    REQUIRE(metrics.averageErrorAfterWarmupPx < kCenterConvergenceAverageBudgetPx);
    REQUIRE(metrics.finalErrorPx < kCenterConvergenceFinalBudgetPx);
    REQUIRE(metrics.maxErrorAfterWarmupPx < kCenterConvergenceMaxBudgetPx);
}

void testNegativeCalibratedCountGainsDoNotInvertFeedback()
{
    configureSyntheticTracker();
    config.target_stream_sharpness = Config::kTargetStreamSharpnessDefault;
    config.target_max_pixel_speed = 20000.0f;
    config.target_calibrated_pixel_counts_enabled = true;
    config.target_counts_per_pixel_x = -1.0f;
    config.target_counts_per_pixel_y = -1.0f;

    LockedTargetInfo locked;
    locked.trackId = 1;
    locked.observedThisFrame = true;
    locked.target.trackId = 1;
    locked.target.smoothX = static_cast<double>(kScreenWidth) * 0.5 + 42.0;
    locked.target.smoothY = static_cast<double>(kScreenHeight) * 0.5;
    locked.target.confidence = 1.0;

    double cameraOffsetX = 0.0;
    double cameraOffsetY = 0.0;
    double countCarryX = 0.0;
    double countCarryY = 0.0;
    const double initialError = locked.target.smoothX - static_cast<double>(kScreenWidth) * 0.5;
    const double moved = applySyntheticStreamTicks(
        locked,
        cameraOffsetX,
        cameraOffsetY,
        countCarryX,
        countCarryY);
    const double finalError = locked.target.smoothX - cameraOffsetX - static_cast<double>(kScreenWidth) * 0.5;

    std::cout << "Synthetic count-sign metrics: initial_error=" << initialError
        << " final_error=" << finalError
        << " camera_offset_x=" << cameraOffsetX
        << " movement_px=" << moved << "\n";

    REQUIRE(moved > 0.0);
    REQUIRE(cameraOffsetX > 0.0);
    REQUIRE(std::abs(finalError) < std::abs(initialError));
}
}

int main()
{
    return obs2test::runTests(
        {
            { "Unreal-style scenario maintains target lock", testUnrealStyleScenarioMaintainsTargetLock },
            { "Unreal-style scenario keeps aim error bounded", testUnrealStyleScenarioKeepsAimErrorBounded },
            { "Unreal-style closed loop converges to center", testUnrealStyleClosedLoopConvergesToCenter },
            { "negative calibrated count gains do not invert feedback", testNegativeCalibratedCountGainsDoNotInvertFeedback },
        },
        "Unreal-style synthetic targeting");
}
