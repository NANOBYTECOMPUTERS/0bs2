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

#include "mouse.h"
#include "OnnxPidGovernor.h"
#include "capture.h"
#include "0BS_box_2.h"

namespace
{
aim::AimKalmanSettings buildKalmanSettingsFromConfig()
{
    aim::AimKalmanSettings settings;
    settings.enabled = config.kalman_enabled;
    settings.process_noise_position = static_cast<double>(config.kalman_process_noise_position);
    settings.process_noise_velocity = static_cast<double>(config.kalman_process_noise_velocity);
    settings.measurement_noise = static_cast<double>(config.kalman_measurement_noise);
    settings.velocity_damping = static_cast<double>(config.kalman_velocity_damping);
    settings.max_velocity = static_cast<double>(config.kalman_max_velocity);
    settings.warmup_frames = config.kalman_warmup_frames;
    return settings;
}

aim::PidMouseSettings buildPidMouseSettingsFromConfig()
{
    aim::PidMouseSettings settings;
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
    settings.feedForwardMaxStep = static_cast<double>(config.pid_feed_forward_max_step);
    settings.feedForwardMinSpeed = static_cast<double>(config.pid_feed_forward_min_speed);
    settings.feedForwardConfidenceFloor = static_cast<double>(config.pid_feed_forward_confidence_floor);
    settings.governorEnabled = config.pid_governor_enabled;
    settings.governorBlend = static_cast<double>(config.pid_governor_blend);
    settings.governorMaxSpeedMultiple = static_cast<double>(config.pid_governor_max_speed_multiple);
    return settings;
}

std::shared_ptr<aim::IPidGovernor> buildPidGovernorFromConfig()
{
    if (!config.pid_governor_enabled)
        return nullptr;
    return aim::createOnnxPidGovernor(config.pid_governor_model_path);
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
    pidController.setSettings(buildPidMouseSettingsFromConfig());
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
        pidController.setSettings(buildPidMouseSettingsFromConfig());
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

            if (command.active)
            {
                const auto counts = pixelDeltaToCounts(command.pixelDx, command.pixelDy);
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
        std::lock_guard<std::mutex> cfgLock(configMutex);
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

    if (mouseInput && mouseInput->move(dx, dy))
        return;

    INPUT in{ 0 };
    in.type = INPUT_MOUSE;
    in.mi.dx = dx;  in.mi.dy = dy;
    in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
    SendInput(1, &in, sizeof(INPUT));
}

void MouseThread::moveRelative(int dx, int dy)
{
    sendMovementToDriver(dx, dy);
}

namespace
{
bool sendWin32LeftButton(DWORD flag)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = flag;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}
}

void MouseThread::pressMouse(const BoxTarget& target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        if (!(mouseInput && mouseInput->leftDown()))
            sendWin32LeftButton(MOUSEEVENTF_LEFTDOWN);
        mouse_pressed = true;
    }
    else if (!bScope && mouse_pressed)
    {
        if (!(mouseInput && mouseInput->leftUp()))
            sendWin32LeftButton(MOUSEEVENTF_LEFTUP);
        mouse_pressed = false;
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
    double targetOffsetY)
{
    aim::PidMouseObservation observation;
    observation.x = pivotX;
    observation.y = pivotY;
    observation.width = targetWidth;
    observation.height = targetHeight;
    observation.targetOffsetX = std::isfinite(targetOffsetX) ? targetOffsetX : 0.0;
    observation.targetOffsetY = std::isfinite(targetOffsetY) ? targetOffsetY : 0.0;
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
    double targetOffsetY)
{
    publishPidObservation(pivotX, pivotY, targetWidth, targetHeight, confidence, targetOffsetX, targetOffsetY);
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
        if (!(mouseInput && mouseInput->leftUp()))
            sendWin32LeftButton(MOUSEEVENTF_LEFTUP);
        mouse_pressed = false;
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
    target_detected.store(false);
}

void MouseThread::setTargetDetected(bool detected)
{
    const bool wasDetected = target_detected.exchange(detected);
    if (!detected && wasDetected)
        resetPid();
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
