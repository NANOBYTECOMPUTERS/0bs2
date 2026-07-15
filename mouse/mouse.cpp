#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <random>
#include <thread>

#include "mouse.h"
#include "capture.h"
#include "0BS_box_2.h"

namespace
{
aim::AimKalmanSettings buildKalmanSettingsFromConfig()
{
    aim::AimKalmanSettings settings;
    settings.runtimeLatencySweepEnabled = config.runtime_latency_sweep_enabled;
    settings.enabled = config.kalman_enabled;
    settings.process_noise_position = static_cast<double>(config.kalman_process_noise_position);
    settings.process_noise_velocity = static_cast<double>(config.kalman_process_noise_velocity);
    settings.measurement_noise = static_cast<double>(config.kalman_measurement_noise);
    settings.velocity_damping = static_cast<double>(config.kalman_velocity_damping);
    settings.max_velocity = static_cast<double>(config.kalman_max_velocity);
    settings.warmup_frames = config.kalman_warmup_frames;
    settings.velocitySeedEnabled = config.kalman_velocity_seed_enabled;
    settings.acquisitionFrames = config.kalman_acquisition_frames;
    return settings;
}

aim::EgoMotionSettings buildEgoMotionSettingsFromConfig()
{
    aim::EgoMotionSettings settings;
    settings.enabled = config.ego_motion_compensation_enabled;
    settings.strength = static_cast<double>(config.ego_motion_compensation_strength);
    const double resolutionScale = std::clamp(static_cast<double>(config.detection_resolution) / 640.0, 0.25, 2.0);
    settings.maxShiftPx = static_cast<double>(config.ego_motion_compensation_max_shift_px) * resolutionScale;
    settings.maxAgeMs = config.ego_motion_compensation_max_age_ms;
    return settings;
}

}

MouseThread::MouseThread(
    int resolution,
    bool auto_shoot,
    float bScope_multiplier,
    IMouseInput* mouseInputDevice)
    : screen_width(resolution),
    screen_height(resolution),
    center_x(resolution / 2.0),
    center_y(resolution / 2.0),
    auto_shoot(auto_shoot),
    bScope_multiplier(bScope_multiplier),
    mouseInput(mouseInputDevice),

    prev_velocity_x(0.0),
    prev_velocity_y(0.0),
    prev_x(0.0),
    prev_y(0.0)
{
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();

    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    directMovementTrackId = -1;
    egoMotionCompensator.setSettings(buildEgoMotionSettingsFromConfig());

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

void MouseThread::updateConfig(
    int resolution,
    bool auto_shoot,
    float bScope_multiplier
)
{
    screen_width = screen_height = resolution;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;

    center_x = center_y = resolution / 2.0;

    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    directMovementTrackId = -1;
    resetEgoMotionCompensation();

    resetDirectMovement();
}

void MouseThread::updateDetectionGeometry(int width, int height)
{
    width = std::max(1, width);
    height = std::max(1, height);

    if (std::abs(screen_width - static_cast<double>(width)) < 0.5 &&
        std::abs(screen_height - static_cast<double>(height)) < 0.5)
    {
        return;
    }

    screen_width = static_cast<double>(width);
    screen_height = static_cast<double>(height);
    center_x = screen_width * 0.5;
    center_y = screen_height * 0.5;

    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    directMovementTrackId = -1;
    resetEgoMotionCompensation();

    resetDirectMovement();
}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::queueMove(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard lg(queueMtx);
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx,dy });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    try
    {
        while (!workerStop)
        {
            std::unique_lock ul(queueMtx);
            queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });

            while (!moveQueue.empty())
            {
                Move m = moveQueue.front();
                moveQueue.pop();
                ul.unlock();
                sendMovementToDriver(m.dx, m.dy);
                appendWindDebugStep(m.dx, m.dy);
                ul.lock();
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Move worker crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Move worker crashed: unknown exception." << std::endl;
    }
}

void MouseThread::resetWindState()
{
    constexpr double twoPi = 6.28318530717958647692;
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> rateDist(0.04, 0.16);

    windCarryX = 0.0;
    windCarryY = 0.0;
    windVelX = 0.0;
    windVelY = 0.0;
    windNoiseX = 0.0;
    windNoiseY = 0.0;
    windFracX = 0.0;
    windFracY = 0.0;
    windPatternX = 0.0;
    windPatternY = 0.0;
    windPatternPhaseA = phaseDist(windRng);
    windPatternPhaseB = phaseDist(windRng);
    windPatternRateA = rateDist(windRng);
    windPatternRateB = rateDist(windRng);
}

void MouseThread::appendWindDebugStep(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    double deltaPxX = static_cast<double>(dx);
    double deltaPxY = static_cast<double>(dy);

    {
        std::lock_guard<std::recursive_mutex> cfgLock(configMutex);
        const double countsPerPixelX = static_cast<double>(config.target_counts_per_pixel_x);
        const double countsPerPixelY = static_cast<double>(config.target_counts_per_pixel_y);
        if (config.target_calibrated_pixel_counts_enabled &&
            std::isfinite(countsPerPixelX) &&
            std::isfinite(countsPerPixelY) &&
            std::abs(countsPerPixelX) > 1e-9 &&
            std::abs(countsPerPixelY) > 1e-9)
        {
            deltaPxX = static_cast<double>(dx) / countsPerPixelX;
            deltaPxY = static_cast<double>(dy) / countsPerPixelY;
        }
    }

    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    if (windDebugTrail.empty())
    {
        windDebugCursorX = center_x;
        windDebugCursorY = center_y;
        windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });
    }

    windDebugCursorX += deltaPxX;
    windDebugCursorY += deltaPxY;
    windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });

    constexpr size_t maxTrailPoints = 220;
    while (windDebugTrail.size() > maxTrailPoints)
        windDebugTrail.pop_front();
}

void MouseThread::pruneWindDebugTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto windTrailLifetime = std::chrono::milliseconds(900);
    while (!windDebugTrail.empty() && (now - windDebugTrail.front().t) > windTrailLifetime)
        windDebugTrail.pop_front();
}

double MouseThread::currentDetectionDelaySec() const
{
    double detectionDelaySec = 0.05;
    if (config.backend == "DML")
    {
        if (dml_detector)
            detectionDelaySec = dml_detector->lastInferenceTimeDML.count() * 0.001;
    }
#ifdef USE_CUDA
    else
    {
        detectionDelaySec = trt_detector.lastInferenceTime.count() * 0.001;
    }
#endif
    if (!std::isfinite(detectionDelaySec))
        detectionDelaySec = 0.05;
    return std::clamp(detectionDelaySec, 0.0, 0.2);
}

double MouseThread::currentPredictionLookaheadSec(double detectionDelaySec) const
{
    double lookahead = 0.0;
    if (config.kalman_compensate_detection_delay)
        lookahead += std::max(0.0, detectionDelaySec);
    lookahead += static_cast<double>(config.kalman_additional_prediction_ms) * 0.001;
    return std::clamp(lookahead, 0.0, 1.5);
}

std::pair<double, double> MouseThread::predict_target_position(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());

    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load())
    {
        prev_time = current_time;
        prev_x = target_x;
        prev_y = target_y;
        prev_velocity_x = 0.0;
        prev_velocity_y = 0.0;
        targetKalman.reset();
        lastKalmanTelemetry = targetKalman.update(target_x, target_y, 1.0 / 120.0, 0.0);
        lastPredictionLookaheadSec = 0.0;
        return { target_x, target_y };
    }

    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    if (dt < 1e-8) dt = 1e-8;

    prev_time = current_time;
    prev_x = target_x;
    prev_y = target_y;

    const double detectionDelaySec = currentDetectionDelaySec();
    const double lookaheadSec = currentPredictionLookaheadSec(detectionDelaySec);
    lastPredictionLookaheadSec = lookaheadSec;

    lastKalmanTelemetry = targetKalman.update(target_x, target_y, dt, lookaheadSec);
    prev_velocity_x = lastKalmanTelemetry.velocity_x;
    prev_velocity_y = lastKalmanTelemetry.velocity_y;

    double predictedX = lastKalmanTelemetry.predicted_x;
    double predictedY = lastKalmanTelemetry.predicted_y;
    if (!std::isfinite(predictedX)) predictedX = target_x;
    if (!std::isfinite(predictedY)) predictedY = target_y;

    return { predictedX, predictedY };
}

std::pair<double, double> MouseThread::blendPredictedAimPoint(double pivotX, double pivotY, double confidence)
{
    if (!std::isfinite(pivotX) || !std::isfinite(pivotY))
        return { pivotX, pivotY };

    if (!config.kalman_enabled)
        return { pivotX, pivotY };

    target_detected.store(true);
    const auto predicted = predict_target_position(pivotX, pivotY);
    if (!std::isfinite(predicted.first) || !std::isfinite(predicted.second))
        return { pivotX, pivotY };

    double leadX = predicted.first - pivotX;
    double leadY = predicted.second - pivotY;
    const double maxLead = std::clamp(static_cast<double>(config.target_prediction_max_lead_px), 0.0, 40.0);
    if (maxLead <= 1e-6)
        return { pivotX, pivotY };

    const double leadMagnitude = std::hypot(leadX, leadY);
    if (leadMagnitude > maxLead && leadMagnitude > 1e-9)
    {
        const double scale = maxLead / leadMagnitude;
        leadX *= scale;
        leadY *= scale;
    }

    const double blend = std::clamp(static_cast<double>(config.target_prediction_blend), 0.0, 0.65);
    const double confidenceGate = std::clamp((confidence - 0.30) / 0.55, 0.0, 1.0);
    const double effectiveBlend = blend * confidenceGate;

    return {
        pivotX + leadX * effectiveBlend,
        pivotY + leadY * effectiveBlend
    };
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (mouseInput)
        mouseInput->move(dx, dy);
}

void MouseThread::moveRelative(int dx, int dy)
{
    sendMovementToDriver(dx, dy);
}

void MouseThread::pressMouse(const BoxTarget& target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        if (mouseInput && mouseInput->leftDown())
        {
            mouse_pressed = true;
        }
    }
    else if (!bScope && mouse_pressed)
    {
        if (mouseInput && mouseInput->leftUp())
        {
            mouse_pressed = false;
        }
    }
}

std::pair<double, double> MouseThread::pixelDeltaToCounts(double pixelDx, double pixelDy) const
{
    const double countsPerPixelX = static_cast<double>(config.target_counts_per_pixel_x);
    const double countsPerPixelY = static_cast<double>(config.target_counts_per_pixel_y);
    if (config.target_calibrated_pixel_counts_enabled &&
        std::isfinite(pixelDx) &&
        std::isfinite(pixelDy) &&
        std::isfinite(countsPerPixelX) &&
        std::isfinite(countsPerPixelY) &&
        std::abs(countsPerPixelX) > 1e-9 &&
        std::abs(countsPerPixelY) > 1e-9)
    {
        return { pixelDx * countsPerPixelX, pixelDy * countsPerPixelY };
    }

    return { pixelDx, pixelDy };
}

void MouseThread::recordEgoMotionDelta(double pixelDx, double pixelDy, std::chrono::steady_clock::time_point timestamp)
{
    updateEgoMotionVelocityEstimate(pixelDx, pixelDy, timestamp);
    egoMotionCompensator.setSettings(buildEgoMotionSettingsFromConfig());
    egoMotionCompensator.recordDelta(pixelDx, pixelDy, timestamp);
}

void MouseThread::updateEgoMotionVelocityEstimate(
    double pixelDx,
    double pixelDy,
    std::chrono::steady_clock::time_point timestamp)
{
    if (!std::isfinite(pixelDx) || !std::isfinite(pixelDy))
        return;

    const double movementPx = std::hypot(pixelDx, pixelDy);
    std::lock_guard<std::mutex> lock(egoMotionVelocityMutex);

    double dt = 1.0 / std::clamp(static_cast<double>(config.capture_fps), 30.0, 240.0);
    if (egoMotionVelocityLastTimestamp.time_since_epoch().count() != 0 && timestamp > egoMotionVelocityLastTimestamp)
        dt = std::chrono::duration<double>(timestamp - egoMotionVelocityLastTimestamp).count();
    egoMotionVelocityLastTimestamp = timestamp;

    dt = std::clamp(dt, 1.0 / 2000.0, 0.25);
    const double rawVelocity = std::clamp(movementPx / dt, 0.0, 4000.0);
    constexpr double egoVelocityAlpha = 0.35;
    egoMotionCameraVelocityPxPerSec += (rawVelocity - egoMotionCameraVelocityPxPerSec) * egoVelocityAlpha;
}

double MouseThread::currentEgoMotionCameraVelocityPxPerSec() const
{
    std::lock_guard<std::mutex> lock(egoMotionVelocityMutex);
    if (egoMotionVelocityLastTimestamp.time_since_epoch().count() == 0)
        return 0.0;

    const double ageSec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - egoMotionVelocityLastTimestamp).count();
    if (ageSec > 0.25)
        return egoMotionCameraVelocityPxPerSec * 0.25;
    if (ageSec > 0.10)
        return egoMotionCameraVelocityPxPerSec * 0.65;
    return egoMotionCameraVelocityPxPerSec;
}

std::pair<double, double> MouseThread::consumeEgoMotionCompensation(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    egoMotionCompensator.setSettings(buildEgoMotionSettingsFromConfig());
    const auto shift = egoMotionCompensator.consume(start, end, std::chrono::steady_clock::now());
    if (!shift.valid)
        return { 0.0, 0.0 };
    return { shift.dx, shift.dy };
}

void MouseThread::resetEgoMotionCompensation()
{
    egoMotionCompensator.setSettings(buildEgoMotionSettingsFromConfig());
    egoMotionCompensator.reset();
    std::lock_guard<std::mutex> lock(egoMotionVelocityMutex);
    egoMotionVelocityLastTimestamp = {};
    egoMotionCameraVelocityPxPerSec = 0.0;
}

void MouseThread::dispatchTargetMovement(
    double pivotX,
    double pivotY,
    double targetOffsetX,
    double targetOffsetY)
{
    if (!std::isfinite(pivotX) || !std::isfinite(pivotY))
        return;

    const double aimX = pivotX + (std::isfinite(targetOffsetX) ? targetOffsetX : 0.0);
    const double aimY = pivotY + (std::isfinite(targetOffsetY) ? targetOffsetY : 0.0);
    double pixelDx = aimX - center_x;
    double pixelDy = aimY - center_y;

    if (!std::isfinite(pixelDx) || !std::isfinite(pixelDy))
        return;

    const double distance = std::hypot(pixelDx, pixelDy);
    const double deadzone = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
    if (distance <= deadzone)
    {
        updateLastAppliedMouseDelta(0.0, 0.0);
        return;
    }

    const double outputScale = std::clamp(static_cast<double>(config.target_output_scale), 0.01, 3.0);
    const double maxStep = std::clamp(static_cast<double>(config.target_max_pixel_step), 0.25, 240.0);
    pixelDx *= outputScale;
    pixelDy *= outputScale;

    const double stepLength = std::hypot(pixelDx, pixelDy);
    if (stepLength > maxStep && stepLength > 1e-9)
    {
        const double scale = maxStep / stepLength;
        pixelDx *= scale;
        pixelDy *= scale;
    }

    const auto now = std::chrono::steady_clock::now();
    target_detected.store(true);
    last_target_time = now;
    recordEgoMotionDelta(pixelDx, pixelDy, now);
    updateLastAppliedMouseDelta(pixelDx, pixelDy);

    const auto counts = pixelDeltaToCounts(pixelDx, pixelDy);
    int dx = 0;
    int dy = 0;
    {
        std::lock_guard<std::mutex> lock(movementMtx);
        movementCountCarryX += counts.first;
        movementCountCarryY += counts.second;
        dx = static_cast<int>(std::round(movementCountCarryX));
        dy = static_cast<int>(std::round(movementCountCarryY));
        movementCountCarryX -= static_cast<double>(dx);
        movementCountCarryY -= static_cast<double>(dy);
    }

    if (dx != 0 || dy != 0)
        queueMove(dx, dy);
}

void MouseThread::resetDirectMovement()
{
    std::lock_guard<std::mutex> lock(movementMtx);
    movementCountCarryX = 0.0;
    movementCountCarryY = 0.0;
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;

    double reduced_w = target_w * (reduction_factor / 2.0);
    double reduced_h = target_h * (reduction_factor / 2.0);

    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;

    return (center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2);
}

void MouseThread::moveMouse(const BoxTarget& target)
{
    moveMouseTarget(target);
}

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    moveMousePivot(pivotX, pivotY, 0.0, 0.0, 1.0);
}

void MouseThread::moveMousePivot(double pivotX, double pivotY, double targetWidth, double targetHeight, double confidence)
{
    (void)targetWidth;
    (void)targetHeight;
    const auto blendedAim = blendPredictedAimPoint(pivotX, pivotY, confidence);
    dispatchTargetMovement(blendedAim.first, blendedAim.second);
}

void MouseThread::moveMousePivot(
    double pivotX,
    double pivotY,
    double targetWidth,
    double targetHeight,
    double confidence,
    double targetOffsetX,
    double targetOffsetY)
{
    (void)targetWidth;
    (void)targetHeight;
    const double observedX = pivotX + (std::isfinite(targetOffsetX) ? targetOffsetX : 0.0);
    const double observedY = pivotY + (std::isfinite(targetOffsetY) ? targetOffsetY : 0.0);
    const auto blendedAim = blendPredictedAimPoint(observedX, observedY, confidence);
    dispatchTargetMovement(blendedAim.first, blendedAim.second);
}

void MouseThread::moveMouseTarget(const BoxTarget& target)
{
    if (target.trackId != directMovementTrackId)
    {
        resetPrediction();
        directMovementTrackId = target.trackId;
    }

    moveMousePivot(
        target.smoothX,
        target.smoothY,
        target.w,
        target.h,
        target.confidence);
}

void MouseThread::updateLastAppliedMouseDelta(double dx, double dy)
{
    std::lock_guard<std::mutex> lock(lastAppliedMouseDeltaMutex);
    lastAppliedMouseDelta = { dx, dy };
}

std::pair<double, double> MouseThread::getLastAppliedMouseDelta() const
{
    std::lock_guard<std::mutex> lock(lastAppliedMouseDeltaMutex);
    return lastAppliedMouseDelta;
}

void MouseThread::clearQueuedMoves()
{
    std::lock_guard<std::mutex> lock(queueMtx);
    std::queue<Move> empty;
    moveQueue.swap(empty);
    resetWindState();
    resetEgoMotionCompensation();
    resetDirectMovement();
}

void MouseThread::releaseMouse()
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (mouse_pressed)
    {
        if (mouseInput && mouseInput->leftUp())
        {
            mouse_pressed = false;
        }
    }
}

void MouseThread::resetPrediction()
{
    clearQueuedMoves();
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0;
    prev_y = 0;
    prev_velocity_x = 0;
    prev_velocity_y = 0;
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    directMovementTrackId = -1;
    resetDirectMovement();
    resetEgoMotionCompensation();
    target_detected.store(false);
}

void MouseThread::setTargetDetected(bool detected)
{
    const bool wasDetected = target_detected.exchange(detected);
    if (!detected && wasDetected)
    {
        resetPrediction();
    }
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();
    const double timeoutSec = std::clamp(static_cast<double>(config.kalman_reset_timeout_sec), 0.05, 3.0);

    if (elapsed > timeoutSec && target_detected.load())
    {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    if (frames <= 0)
        return result;

    result.reserve(frames);

    const double fixedFps = 30.0;
    const double frame_time = 1.0 / fixedFps;

    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    if (targetKalman.initialized())
    {
        const double detectionDelaySec = currentDetectionDelaySec();
        const double baseLookaheadSec = currentPredictionLookaheadSec(detectionDelaySec);
        for (int i = 1; i <= frames; ++i)
        {
            const double t = baseLookaheadSec + frame_time * i;
            auto predicted = targetKalman.predict(t);
            if (!std::isfinite(predicted.first) || !std::isfinite(predicted.second))
                continue;
            result.push_back(predicted);
        }

        if (!result.empty())
            return result;
    }

    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();

    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5)
    {
        return result;
    }

    double vx = prev_velocity_x;
    double vy = prev_velocity_y;
    
    for (int i = 1; i <= frames; i++)
    {
        double t = frame_time * i;
        double px = pivotX + vx * t;
        double py = pivotY + vy * t;
        result.push_back({ px, py });
    }

    return result;
}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::clearWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    windDebugTrail.clear();
    windDebugCursorX = center_x;
    windDebugCursorY = center_y;
}

std::vector<std::pair<double, double>> MouseThread::getWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    std::vector<std::pair<double, double>> out;
    out.reserve(windDebugTrail.size());
    for (const auto& p : windDebugTrail)
        out.emplace_back(p.x, p.y);
    return out;
}

void MouseThread::setMouseInput(IMouseInput* newMouseInput)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    mouseInput = newMouseInput;
}
