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
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "mouse.h"
#include "OnnxPidGovernor.h"
#include "capture.h"
#include "0BS_box_2.h"

namespace
{
constexpr double AdaptiveInfluenceActivateThreshold = 0.025;
constexpr double AdaptiveInfluenceDeactivateThreshold = 0.010;

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

aim::PidMouseSettings buildPidMouseSettingsFromConfig(
    double screen_width,
    double screen_height,
    double fov_x,
    double fov_y)
{
    aim::PidMouseSettings settings;
    settings.runtimeLatencySweepEnabled = config.runtime_latency_sweep_enabled;
    settings.actuatorHz = config.pid_actuator_hz;
    settings.kp = static_cast<double>(config.pid_kp);
    settings.ki = static_cast<double>(config.pid_ki);
    settings.kd = static_cast<double>(config.pid_kd);
    settings.deadzonePx = static_cast<double>(config.pid_deadzone_px);
    settings.maxPixelStep = static_cast<double>(config.pid_max_pixel_step);
    settings.outputScale = static_cast<double>(config.pid_output_scale);
    settings.minOutputScale = static_cast<double>(config.pid_min_output_scale);
    settings.maxOutputScale = static_cast<double>(config.pid_max_output_scale);
    settings.sizeReferencePx = static_cast<double>(config.pid_size_reference_px);
    settings.sizeMinScale = static_cast<double>(config.pid_size_min_scale);
    settings.sizeMaxScale = static_cast<double>(config.pid_size_max_scale);
    settings.precisionRadiusScale = static_cast<double>(config.pid_precision_radius_scale);
    settings.slowdownRadiusScale = static_cast<double>(config.pid_slowdown_radius_scale);
    settings.overshootBrake = static_cast<double>(config.pid_overshoot_brake);
    settings.divergenceBoost = static_cast<double>(config.pid_divergence_boost);
    settings.scaleResponse = static_cast<double>(config.pid_scale_response);
    settings.maxIntegral = static_cast<double>(config.pid_max_integral);
    settings.maxDerivativeTerm = static_cast<double>(config.pid_max_derivative_term);
    settings.derivativeFilterTauMs = static_cast<double>(config.pid_derivative_filter_tau_ms);
    settings.targetLossTimeoutMs = static_cast<double>(config.pid_target_loss_timeout_ms);
    settings.feedForwardEnabled = config.pid_feed_forward_enabled;
    settings.feedForwardGain = static_cast<double>(config.pid_feed_forward_gain);
    settings.feedForwardLookaheadMs = static_cast<double>(config.pid_feed_forward_lookahead_ms);
    settings.feedForwardFrameLookahead = config.pid_feed_forward_frame_lookahead;
    settings.feedForwardMaxStep = static_cast<double>(config.pid_feed_forward_max_step);
    settings.feedForwardMinSpeed = static_cast<double>(config.pid_feed_forward_min_speed);
    settings.feedForwardConfidenceFloor = static_cast<double>(config.pid_feed_forward_confidence_floor);
    settings.conditionalIntegrationEnabled = config.pid_conditional_integration_enabled;
    settings.conditionalIntegrationErrorPx = static_cast<double>(config.pid_conditional_integration_error_px);
    settings.adaptiveOutputScalingEnabled = config.pid_adaptive_output_scaling_enabled;
    settings.adaptiveOutputErrorScale = static_cast<double>(config.pid_adaptive_output_error_scale);
    settings.derivativeSmoothingMultiplier = static_cast<double>(config.pid_derivative_smoothing_multiplier);
    settings.perspectiveFovMappingEnabled = config.pid_perspective_fov_mapping_enabled;
    settings.projectionWidthPx = screen_width;
    settings.projectionHeightPx = screen_height;
    settings.fovXDeg = fov_x;
    settings.fovYDeg = fov_y;
    settings.governorEnabled = config.pid_governor_enabled;
    settings.governorBlend = static_cast<double>(config.pid_governor_blend);
    settings.governorMaxSpeedMultiple = static_cast<double>(config.pid_governor_max_speed_multiple);
    settings.pid_smart_blending_enabled = config.pid_smart_blending_enabled;
    settings.pid_smart_blending_aggression = static_cast<double>(config.pid_smart_blending_aggression);
    settings.pid_smart_blending_near_damping = static_cast<double>(config.pid_smart_blending_near_damping);
    settings.pid_smart_blending_deadzone_px = static_cast<double>(config.pid_smart_blending_deadzone_px);
    settings.pid_smart_blending_jerk_limit_px = static_cast<double>(config.pid_smart_blending_jerk_limit_px);
    settings.pid_smart_blending_confidence_floor = static_cast<double>(config.pid_smart_blending_confidence_floor);
    return settings;
}

std::shared_ptr<aim::IPidGovernor> buildPidGovernorFromConfig()
{
    if (!config.pid_governor_enabled)
        return nullptr;
    return aim::createOnnxPidGovernor(config.pid_governor_model_path);
}

double smoothStep01(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

double smoothStepRange(double edge0, double edge1, double value)
{
    if (edge1 <= edge0)
        return value >= edge1 ? 1.0 : 0.0;

    return smoothStep01((value - edge0) / (edge1 - edge0));
}

std::pair<double, double> clampVectorLength(double x, double y, double maxLength)
{
    const double length = std::hypot(x, y);
    if (!std::isfinite(length) || length <= maxLength || length <= 1e-9)
        return { x, y };

    const double scale = maxLength / length;
    return { x * scale, y * scale };
}
}

MouseThread::MouseThread(
    int resolution,
    int fovX,
    int fovY,
    bool auto_shoot,
    float bScope_multiplier,
    IMouseInput* mouseInputDevice)
    : screen_width(resolution),
    screen_height(resolution),
    fov_x(fovX),
    fov_y(fovY),
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
    pidController.setSettings(buildPidMouseSettingsFromConfig(screen_width, screen_height, fov_x, fov_y));
    pidController.setGovernor(buildPidGovernorFromConfig());

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
    pidActuator = std::thread(&MouseThread::pidActuatorLoop, this);
}

void MouseThread::updateConfig(
    int resolution,
    int fovX,
    int fovY,
    bool auto_shoot,
    float bScope_multiplier
)
{
    screen_width = screen_height = resolution;
    fov_x = fovX;  fov_y = fovY;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;

    center_x = center_y = resolution / 2.0;

    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;

    {
        std::lock_guard<std::mutex> lock(pidMtx);
        pidController.setSettings(buildPidMouseSettingsFromConfig(screen_width, screen_height, fov_x, fov_y));
        pidController.setGovernor(buildPidGovernorFromConfig());
        pidController.reset();
        pidCountCarryX = 0.0;
        pidCountCarryY = 0.0;
    }
}

MouseThread::~MouseThread()
{
    pidStop = true;
    if (pidActuator.joinable()) pidActuator.join();

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

void MouseThread::pidActuatorLoop()
{
    auto nextWake = std::chrono::steady_clock::now();

    try
    {
        while (!pidStop)
        {
            const auto now = std::chrono::steady_clock::now();
            aim::PidMouseCommand command;
            double actuatorHz = 240.0;

            {
                std::lock_guard<std::mutex> lock(pidMtx);
                const auto& settings = pidController.getSettings();
                actuatorHz = std::clamp(static_cast<double>(settings.actuatorHz), 30.0, 2000.0);
                command = pidController.step(now);
            }
            updateNeuralControlActuatorTelemetry(command);

            if (command.active)
            {
                std::pair<double, double> counts{ 0.0, 0.0 };
                if (command.angularOutputActive)
                {
                    try
                    {
                        counts = config.degToCounts(command.angularDxDeg, command.angularDyDeg, fov_x);
                    }
                    catch (...)
                    {
                        counts = { 0.0, 0.0 };
                    }
                }
                else
                {
                    counts = pixelDeltaToCounts(command.pixelDx, command.pixelDy);
                }
                int dx = 0;
                int dy = 0;
                {
                    std::lock_guard<std::mutex> lock(pidMtx);
                    pidCountCarryX += counts.first;
                    pidCountCarryY += counts.second;

                    dx = static_cast<int>(std::round(pidCountCarryX));
                    dy = static_cast<int>(std::round(pidCountCarryY));
                    pidCountCarryX -= static_cast<double>(dx);
                    pidCountCarryY -= static_cast<double>(dy);
                }

                if (dx != 0 || dy != 0)
                    queueMove(dx, dy);
            }

            const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / actuatorHz));
            nextWake += period;

            const auto afterStep = std::chrono::steady_clock::now();
            if (nextWake <= afterStep)
                nextWake = afterStep + period;

            std::this_thread::sleep_until(nextWake);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] PID actuator crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] PID actuator crashed: unknown exception." << std::endl;
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
        const Config::GameProfile* gpPtr = nullptr;

        auto activeIt = config.game_profiles.find(config.active_game);
        if (activeIt != config.game_profiles.end())
            gpPtr = &activeIt->second;
        else
        {
            auto unifiedIt = config.game_profiles.find("UNIFIED");
            if (unifiedIt != config.game_profiles.end())
                gpPtr = &unifiedIt->second;
        }

        if (gpPtr && gpPtr->sens != 0.0 && gpPtr->yaw != 0.0 && gpPtr->pitch != 0.0)
        {
            const double fovNow = std::max(1.0, fov_x);
            const double fovScale = (gpPtr->fovScaled && gpPtr->baseFOV > 1.0) ? (fovNow / gpPtr->baseFOV) : 1.0;
            const double degX = static_cast<double>(dx) * gpPtr->sens * gpPtr->yaw * fovScale;
            const double degY = static_cast<double>(dy) * gpPtr->sens * gpPtr->pitch * fovScale;

            const double degPerPxX = fov_x / std::max(1.0, screen_width);
            const double degPerPxY = fov_y / std::max(1.0, screen_height);

            if (std::abs(degPerPxX) > 1e-8 && std::abs(degPerPxY) > 1e-8)
            {
                deltaPxX = degX / degPerPxX;
                deltaPxY = degY / degPerPxY;
            }
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
    const double degPerPxX = fov_x / std::max(1.0, screen_width);
    const double degPerPxY = fov_y / std::max(1.0, screen_height);
    const double degX = pixelDx * degPerPxX;
    const double degY = pixelDy * degPerPxY;

    try
    {
        return config.degToCounts(degX, degY, fov_x);
    }
    catch (...)
    {
        return { 0.0, 0.0 };
    }
}

void MouseThread::publishPidObservation(
    double pivotX,
    double pivotY,
    double targetWidth,
    double targetHeight,
    double confidence,
    double targetOffsetX,
    double targetOffsetY,
    double learnedPredictionLeadX,
    double learnedPredictionLeadY)
{
    aim::PidMouseObservation observation;
    observation.x = pivotX;
    observation.y = pivotY;
    observation.width = targetWidth;
    observation.height = targetHeight;
    observation.targetOffsetX = std::isfinite(targetOffsetX) ? targetOffsetX : 0.0;
    observation.targetOffsetY = std::isfinite(targetOffsetY) ? targetOffsetY : 0.0;
    observation.learnedPredictionLeadX = std::isfinite(learnedPredictionLeadX) ? learnedPredictionLeadX : 0.0;
    observation.learnedPredictionLeadY = std::isfinite(learnedPredictionLeadY) ? learnedPredictionLeadY : 0.0;
    observation.learnedPredictionLeadValid = std::hypot(observation.learnedPredictionLeadX, observation.learnedPredictionLeadY) > 1e-6;
    observation.confidence = std::isfinite(confidence) ? std::clamp(confidence, 0.0, 1.0) : 0.0;
    observation.timestamp = std::chrono::steady_clock::now();
    observation.valid = true;
    target_detected.store(true);
    last_target_time = observation.timestamp;

    std::lock_guard<std::mutex> lock(pidMtx);
    pidController.updateObservation(observation, center_x, center_y);
}

void MouseThread::resetPid()
{
    std::lock_guard<std::mutex> lock(pidMtx);
    pidController.reset();
    pidCountCarryX = 0.0;
    pidCountCarryY = 0.0;
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
    moveMousePivot(
        target.x + target.w / 2.0,
        target.y + target.h / 2.0,
        target.w,
        target.h,
        1.0);
}

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    moveMousePivot(pivotX, pivotY, 0.0, 0.0, 1.0);
}

void MouseThread::moveMousePivot(double pivotX, double pivotY, double targetWidth, double targetHeight, double confidence)
{
    publishPidObservation(pivotX, pivotY, targetWidth, targetHeight, confidence);
}

void MouseThread::moveMousePivot(
    double pivotX,
    double pivotY,
    double targetWidth,
    double targetHeight,
    double confidence,
    double targetOffsetX,
    double targetOffsetY,
    double learnedPredictionLeadX,
    double learnedPredictionLeadY)
{
    publishPidObservation(
        pivotX,
        pivotY,
        targetWidth,
        targetHeight,
        confidence,
        targetOffsetX,
        targetOffsetY,
        learnedPredictionLeadX,
        learnedPredictionLeadY);
}

aim::neural::NeuralTargetingHead::Input MouseThread::computeNeuralTargetingInput(
    const BoxTarget& target,
    const LockedTargetInfo& lockInfo) const
{
    aim::neural::NeuralTargetingHead::Input input;
    input.center_x = static_cast<float>(target.smoothX);
    input.center_y = static_cast<float>(target.smoothY);
    input.width = static_cast<float>(target.w);
    input.height = static_cast<float>(target.h);
    input.vx = static_cast<float>(lockInfo.targetVelocityX);
    input.vy = static_cast<float>(lockInfo.targetVelocityY);
    input.box_scale_vel = static_cast<float>(lockInfo.targetBoxScaleVelocity);
    input.confidence = static_cast<float>(std::clamp(target.confidence, 0.0, 1.0));

    input.predicted_x.reserve(lockInfo.predictedFuture.size());
    input.predicted_y.reserve(lockInfo.predictedFuture.size());
    for (const auto& p : lockInfo.predictedFuture)
    {
        if (!std::isfinite(p.first) || !std::isfinite(p.second))
            continue;
        input.predicted_x.push_back(static_cast<float>(p.first));
        input.predicted_y.push_back(static_cast<float>(p.second));
    }

    return input;
}

std::pair<double, double> MouseThread::consumeNeuralTargetingResult(
    const BoxTarget& target,
    const LockedTargetInfo& lockInfo)
{
    if (!config.neural_targeting_enabled)
        return { 0.0, 0.0 };

    aim::neural::NeuralTargetingWorker::Result targetingResult;
    if (!aim::neural::NeuralTargetingWorker::instance().tryGet(lockInfo.trackId, targetingResult))
        return neuralTargetingRefinementValid && neuralTargetingLastResultTrackId == lockInfo.trackId
            ? neuralTargetingRefinement
            : std::pair<double, double>{ 0.0, 0.0 };

    neuralTargetingPending = false;
    neuralTargetingPendingTrackId = -1;

    const bool highResolution = config.detection_resolution >= 640;
    const double confidenceFloor = highResolution ? 0.60 : 0.50;
    if (!targetingResult.valid ||
        targetingResult.track_id != lockInfo.trackId ||
        targetingResult.output.confidence < confidenceFloor)
    {
        neuralTargetingRefinementValid = false;
        neuralTargetingRefinement = { 0.0, 0.0 };
        return { 0.0, 0.0 };
    }

    const double maxRefinement = std::clamp(
        static_cast<double>(config.neural_targeting_max_refinement_px),
        1.0,
        80.0);
    const auto clamped = clampVectorLength(
        targetingResult.output.refinement_offset_x,
        targetingResult.output.refinement_offset_y,
        maxRefinement);
    std::pair<double, double> neuralRefinement = clamped;
    if (targetingResult.output.confidence < 0.85)
    {
        neuralRefinement = rejectNeuralRefinementAgainstPidDirection(
            clamped,
            target.smoothX - center_x,
            target.smoothY - center_y,
            targetingResult.output.confidence);
    }
    const double influence = std::clamp(static_cast<double>(config.neural_targeting_influence), 0.0, 1.0);

    neuralTargetingRefinement = {
        neuralRefinement.first * influence,
        neuralRefinement.second * influence
    };
    neuralTargetingLastResultTrackId = lockInfo.trackId;
    neuralTargetingRefinementValid =
        std::isfinite(neuralTargetingRefinement.first) &&
        std::isfinite(neuralTargetingRefinement.second) &&
        std::hypot(neuralTargetingRefinement.first, neuralTargetingRefinement.second) > 1e-6;

    if (!neuralTargetingRefinementValid)
        neuralTargetingRefinement = { 0.0, 0.0 };

    return neuralTargetingRefinement;
}

void MouseThread::submitNeuralTargetingRequest(
    const BoxTarget& target,
    const LockedTargetInfo& lockInfo)
{
    if (!config.neural_targeting_enabled)
    {
        aim::neural::NeuralTargetingWorker::instance().clear();
        neuralTargetingPending = false;
        neuralTargetingPendingTrackId = -1;
        neuralTargetingRefinementValid = false;
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (neuralTargetingPending &&
        neuralTargetingPendingTrackId == lockInfo.trackId &&
        now - neuralTargetingLastSubmit < std::chrono::milliseconds(33))
    {
        return;
    }

    aim::neural::NeuralTargetingWorker::Request request;
    request.track_id = lockInfo.trackId;
    request.frame_id = lockInfo.predictedFutureAgeFrames;
    request.model_path = config.neural_targeting_model_path;
    request.max_refinement_px = std::clamp(config.neural_targeting_max_refinement_px, 1.0f, 80.0f);
    request.max_iterations = std::clamp(config.neural_targeting_max_iterations, 1, 2);
    request.input = computeNeuralTargetingInput(target, lockInfo);
    aim::neural::NeuralTargetingWorker::instance().submit(request);

    neuralTargetingPending = true;
    neuralTargetingPendingTrackId = lockInfo.trackId;
    neuralTargetingLastSubmit = now;
}

void MouseThread::setNeuralTargetingDebugPoint(const BoxTarget& target, const std::pair<double, double>& totalLead)
{
    std::lock_guard<std::mutex> lock(neuralTargetingDebugMutex);
    neuralTargetingRefinedAimPoint = {
        target.smoothX + totalLead.first,
        target.smoothY + totalLead.second
    };
    neuralTargetingRefinedAimPointValid =
        config.neural_targeting_enabled &&
        std::isfinite(neuralTargetingRefinedAimPoint.first) &&
        std::isfinite(neuralTargetingRefinedAimPoint.second);
}

std::pair<double, double> MouseThread::rejectNeuralRefinementAgainstPidDirection(
    const std::pair<double, double>& neuralRefinement,
    double errorX,
    double errorY,
    double modelConfidence) const
{
    const double neuralMagnitude = std::hypot(neuralRefinement.first, neuralRefinement.second);
    const double errorMagnitude = std::hypot(errorX, errorY);
    if (neuralMagnitude <= 1e-6 || errorMagnitude <= 1e-6)
        return neuralRefinement;

    const double pidDirectionCosine =
        (neuralRefinement.first * errorX + neuralRefinement.second * errorY) /
        std::max(1e-6, neuralMagnitude * errorMagnitude);
    if (pidDirectionCosine < -0.25 && modelConfidence < 0.85)
        return { 0.0, 0.0 };

    return neuralRefinement;
}

double MouseThread::computeAdaptivePredictionInfluence(
    int trackId,
    double distanceToCrosshair,
    double measuredSpeed,
    double directionCosine,
    double confidence,
    int predictionAgeFrames,
    double baseInfluence)
{
    if (!config.temporal_prediction_adaptive_influence_enabled)
        return baseInfluence;

    if (trackId != adaptivePredictionTrackId)
    {
        adaptivePredictionTrackId = trackId;
        adaptivePredictionInfluenceEma = 0.0;
        adaptivePredictionInfluenceActive = false;
    }

    if (predictionAgeFrames > 5)
    {
        adaptivePredictionInfluenceEma *= 0.35;
        return std::clamp(adaptivePredictionInfluenceEma, 0.0, 0.85);
    }

    const double distanceBand =
        smoothStepRange(35.0, 90.0, distanceToCrosshair) *
        (1.0 - smoothStepRange(180.0, 320.0, distanceToCrosshair));
    const double adaptiveSpeedFactor = std::clamp(measuredSpeed / 1000.0, 0.0, 1.0);
    const double direction_factor = (directionCosine < -0.35 && confidence < 0.82) ? 0.30 : 1.0;
    const double age_factor = (predictionAgeFrames <= 3) ? 1.0 : 0.35;
    double rawInfluence =
        baseInfluence * distanceBand * adaptiveSpeedFactor * confidence * direction_factor * age_factor;
    rawInfluence = std::clamp(rawInfluence, 0.0, 0.85);
    if (!adaptivePredictionInfluenceActive && rawInfluence >= AdaptiveInfluenceActivateThreshold)
        adaptivePredictionInfluenceActive = true;
    else if (adaptivePredictionInfluenceActive && rawInfluence <= AdaptiveInfluenceDeactivateThreshold)
        adaptivePredictionInfluenceActive = false;

    if (!adaptivePredictionInfluenceActive)
        rawInfluence = 0.0;

    const double emaAlpha = std::clamp(
        static_cast<double>(config.temporal_prediction_adaptive_ema_alpha),
        0.05,
        1.0);

    adaptivePredictionInfluenceEma += (rawInfluence - adaptivePredictionInfluenceEma) * emaAlpha;
    return std::clamp(adaptivePredictionInfluenceEma, 0.0, 0.85);
}

void MouseThread::resetAdaptivePredictionInfluence()
{
    adaptivePredictionInfluenceEma = 0.0;
    adaptivePredictionTrackId = -1;
    adaptivePredictionInfluenceActive = false;
    std::lock_guard<std::mutex> lock(neuralControlTelemetryMutex);
    neuralControlTelemetry.valid = false;
}

void MouseThread::publishNeuralControlTelemetry(
    int trackId,
    double adaptiveInfluence,
    double confidence,
    const std::pair<double, double>& phase2Lead,
    const std::pair<double, double>& neuralRefinement,
    const std::pair<double, double>& totalLead)
{
    NeuralControlTelemetry next;
    next.trackId = trackId;
    next.adaptiveInfluence = std::isfinite(adaptiveInfluence) ? adaptiveInfluence : 0.0;
    next.confidence = std::isfinite(confidence) ? confidence : 0.0;
    next.predictionLeadX = std::isfinite(phase2Lead.first) ? phase2Lead.first : 0.0;
    next.predictionLeadY = std::isfinite(phase2Lead.second) ? phase2Lead.second : 0.0;
    next.neuralRefinementX = std::isfinite(neuralRefinement.first) ? neuralRefinement.first : 0.0;
    next.neuralRefinementY = std::isfinite(neuralRefinement.second) ? neuralRefinement.second : 0.0;
    next.totalLeadX = std::isfinite(totalLead.first) ? totalLead.first : 0.0;
    next.totalLeadY = std::isfinite(totalLead.second) ? totalLead.second : 0.0;
    next.valid = true;

    {
        std::lock_guard<std::mutex> lock(neuralControlTelemetryMutex);
        next.jitterScore = neuralControlTelemetry.jitterScore;
        next.oscillationPenalty = neuralControlTelemetry.oscillationPenalty;
        neuralControlTelemetry = next;
    }

    if (!config.neural_control_telemetry_logging_enabled)
        return;

    const auto now = std::chrono::steady_clock::now();
    const int intervalMs = std::clamp(config.neural_control_telemetry_log_interval_ms, 50, 5000);
    if (lastNeuralControlTelemetryLog.time_since_epoch().count() != 0 &&
        now - lastNeuralControlTelemetryLog < std::chrono::milliseconds(intervalMs))
    {
        return;
    }
    lastNeuralControlTelemetryLog = now;

    const std::filesystem::path logPath(
        config.neural_control_telemetry_log_path.empty()
            ? std::string("logs/neural_control_telemetry.csv")
            : config.neural_control_telemetry_log_path);
    const std::filesystem::path parent = logPath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::error_code existsError;
    const bool writeHeader = !std::filesystem::exists(logPath, existsError);
    std::ofstream file(logPath, std::ios::app);
    if (!file.is_open())
        return;

    if (writeHeader)
    {
        file << "track_id,adaptive_influence,confidence,prediction_lead_x,prediction_lead_y,"
             << "neural_refinement_x,neural_refinement_y,total_lead_x,total_lead_y,jitter_score,oscillation_penalty\n";
    }

    file << std::fixed << std::setprecision(5)
         << next.trackId << ','
         << next.adaptiveInfluence << ','
         << next.confidence << ','
         << next.predictionLeadX << ','
         << next.predictionLeadY << ','
         << next.neuralRefinementX << ','
         << next.neuralRefinementY << ','
         << next.totalLeadX << ','
         << next.totalLeadY << ','
         << next.jitterScore << ','
         << next.oscillationPenalty << '\n';
}

void MouseThread::updateNeuralControlActuatorTelemetry(const aim::PidMouseCommand& command)
{
    std::lock_guard<std::mutex> lock(neuralControlTelemetryMutex);
    if (!neuralControlTelemetry.valid)
        return;

    neuralControlTelemetry.jitterScore = command.smartBlendJitterScore;
    neuralControlTelemetry.oscillationPenalty = command.smartBlendOscillationPenalty;
}

std::pair<double, double> MouseThread::computePredictionFeedForwardLead(
    const BoxTarget& target,
    const LockedTargetInfo& lockInfo)
{
    if (!config.temporal_prediction_enabled)
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    if (lockInfo.predictedFuture.empty())
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    if (lockInfo.predictedFutureAgeFrames > 3)
    {
        if (lockInfo.predictedFutureAgeFrames > 5)
            adaptivePredictionInfluenceEma *= 0.35;
        else
            resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    const auto first = lockInfo.predictedFuture.front();
    if (!std::isfinite(first.first) || !std::isfinite(first.second) ||
        !std::isfinite(target.smoothX) || !std::isfinite(target.smoothY))
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    const bool highResolution = config.detection_resolution >= 640;
    const double maxFirstPointDistance = highResolution ? 65.0 : 50.0;
    const double firstPointDistance = std::hypot(first.first - target.smoothX, first.second - target.smoothY);
    if (!std::isfinite(firstPointDistance) || firstPointDistance > maxFirstPointDistance)
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    double predictionVx = first.first - target.smoothX;
    double predictionVy = first.second - target.smoothY;
    if (lockInfo.predictedFuture.size() > 1)
    {
        predictionVx = lockInfo.predictedFuture[1].first - first.first;
        predictionVy = lockInfo.predictedFuture[1].second - first.second;
    }

    const double predictionFps = 60.0;
    const double predictionSpeed = std::hypot(predictionVx, predictionVy) * predictionFps;
    const double maxPredictionSpeed = highResolution ? 1500.0 : 1200.0;
    if (!std::isfinite(predictionSpeed) || predictionSpeed > maxPredictionSpeed)
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    const double confidenceFloor = highResolution ? 0.65 : 0.55;
    const double confidence = std::isfinite(target.confidence) ? std::clamp(target.confidence, 0.0, 1.0) : 0.0;
    if (confidence < confidenceFloor)
    {
        resetAdaptivePredictionInfluence();
        return { 0.0, 0.0 };
    }

    const double measuredVx = lockInfo.targetVelocityX;
    const double measuredVy = lockInfo.targetVelocityY;
    double directionCosine = 1.0;
    const double measuredSpeed = std::hypot(measuredVx, measuredVy);
    if (measuredSpeed > 100.0 && predictionSpeed > 100.0)
    {
        const double predMag = std::hypot(predictionVx, predictionVy);
        directionCosine = (predictionVx * measuredVx + predictionVy * measuredVy) /
            std::max(1e-6, predMag * measuredSpeed);
        if (directionCosine < -0.35 && confidence < 0.85)
        {
            resetAdaptivePredictionInfluence();
            return { 0.0, 0.0 };
        }
    }

    std::pair<double, double> phase2Lead{ 0.0, 0.0 };
    const double distanceToCrosshair = std::hypot(target.smoothX - center_x, target.smoothY - center_y);
    const double near_damping = smoothStep01(distanceToCrosshair / 80.0);
    const double speed_weight = std::clamp(measuredSpeed / 1200.0, 0.0, 1.0);
    const double baseInfluence = std::clamp(static_cast<double>(config.temporal_prediction_influence), 0.0, 1.0);
    const double influence = computeAdaptivePredictionInfluence(
        lockInfo.trackId,
        distanceToCrosshair,
        measuredSpeed,
        directionCosine,
        confidence,
        lockInfo.predictedFutureAgeFrames,
        baseInfluence);
    const double weight = config.temporal_prediction_adaptive_influence_enabled
        ? influence
        : influence * confidence * speed_weight * near_damping;
    if (config.temporal_prediction_feed_forward_enabled && weight > 1e-6)
    {
        const double rawLeadX = first.first - target.smoothX;
        const double rawLeadY = first.second - target.smoothY;
        phase2Lead = clampVectorLength(
            rawLeadX * weight,
            rawLeadY * weight,
            std::clamp(static_cast<double>(config.temporal_prediction_max_lead_px), 20.0, 80.0));
    }

    std::pair<double, double> neuralRefinement = consumeNeuralTargetingResult(target, lockInfo);
    submitNeuralTargetingRequest(target, lockInfo);

    const double phase2Magnitude = std::hypot(phase2Lead.first, phase2Lead.second);
    const double neuralMagnitude = std::hypot(neuralRefinement.first, neuralRefinement.second);
    if (phase2Magnitude > 1e-6 && neuralMagnitude > 1e-6)
    {
        const double disagreementCosine =
            (phase2Lead.first * neuralRefinement.first + phase2Lead.second * neuralRefinement.second) /
            std::max(1e-6, phase2Magnitude * neuralMagnitude);
        if (disagreementCosine < 0.0 && confidence < 0.85)
        {
            neuralRefinement.first *= 0.25;
            neuralRefinement.second *= 0.25;
        }
    }

    const double phase2MaxLead = std::clamp(static_cast<double>(config.temporal_prediction_max_lead_px), 20.0, 80.0);
    const double neuralMaxLead = config.neural_targeting_enabled
        ? std::clamp(static_cast<double>(config.neural_targeting_max_refinement_px), 1.0, 80.0)
        : 0.0;
    const double conservativeMaxTotalLead = highResolution ? 48.0 : 38.0;
    const double maxTotalLead = config.neural_targeting_enabled
        ? std::min(phase2MaxLead + neuralMaxLead, conservativeMaxTotalLead)
        : phase2MaxLead;
    auto total = clampVectorLength(
        phase2Lead.first + neuralRefinement.first,
        phase2Lead.second + neuralRefinement.second,
        maxTotalLead);
    setNeuralTargetingDebugPoint(target, total);
    publishNeuralControlTelemetry(lockInfo.trackId, weight, confidence, phase2Lead, neuralRefinement, total);
    return total;
}

void MouseThread::clearQueuedMoves()
{
    std::lock_guard<std::mutex> lock(queueMtx);
    std::queue<Move> empty;
    moveQueue.swap(empty);
    resetWindState();
    resetPid();
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
    resetAdaptivePredictionInfluence();
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0;
    prev_y = 0;
    prev_velocity_x = 0;
    prev_velocity_y = 0;
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    target_detected.store(false);
}

void MouseThread::setTargetDetected(bool detected)
{
    const bool wasDetected = target_detected.exchange(detected);
    if (!detected && wasDetected)
    {
        resetPid();
        resetAdaptivePredictionInfluence();
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
    resetAdaptivePredictionInfluence();
    {
        std::lock_guard<std::mutex> debugLock(neuralTargetingDebugMutex);
        neuralTargetingRefinedAimPointValid = false;
    }
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

bool MouseThread::getNeuralTargetingRefinedAimPoint(std::pair<double, double>& point) const
{
    std::lock_guard<std::mutex> lock(neuralTargetingDebugMutex);
    if (!neuralTargetingRefinedAimPointValid)
        return false;

    point = neuralTargetingRefinedAimPoint;
    return true;
}

bool MouseThread::getNeuralControlTelemetry(NeuralControlTelemetry& telemetry) const
{
    std::lock_guard<std::mutex> lock(neuralControlTelemetryMutex);
    if (!neuralControlTelemetry.valid)
        return false;

    telemetry = neuralControlTelemetry;
    return true;
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
