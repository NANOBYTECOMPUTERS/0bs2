#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <limits>
#include <sstream>
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
    targetStreamWorker = std::thread(&MouseThread::targetStreamWorkerLoop, this);
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
    clearTargetMotionState();
    if (!config.target_signal_diagnostics_enabled)
        resetTargetSignalDiagnostics();
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
    clearTargetMotionState();
    if (!config.target_signal_diagnostics_enabled)
        resetTargetSignalDiagnostics();
}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    targetStreamCv.notify_all();
    if (targetStreamWorker.joinable()) targetStreamWorker.join();
    if (moveWorker.joinable()) moveWorker.join();
    resetTargetSignalDiagnostics();
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

bool MouseThread::snapshotTargetMotionState(TargetMotionState& out) const
{
    std::lock_guard<std::mutex> lock(targetStateMutex);
    if (!targetMotionState.valid)
        return false;

    out = targetMotionState;
    return true;
}

MouseThread::TargetStreamDebugSnapshot MouseThread::makeTargetStreamDebugSnapshot(
    const TargetMotionState* state,
    std::chrono::steady_clock::time_point now,
    double dtSec,
    const char* status)
{
    TargetStreamDebugSnapshot snapshot;
    snapshot.enabled = config.target_stream_debug_enabled;
    if (!snapshot.enabled && !targetSignalDiagnosticsActive())
        return snapshot;

    snapshot.streamEnabled = config.target_stream_enabled;
    snapshot.aimingActive = aiming.load(std::memory_order_relaxed);
    snapshot.status = status;
    snapshot.tickDtMs = dtSec * 1000.0;
    snapshot.centerX = center_x;
    snapshot.centerY = center_y;
    snapshot.deadzonePx = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
    snapshot.maxSpeedPxPerSec = std::clamp(static_cast<double>(config.target_max_pixel_speed), 50.0, 20000.0);
    snapshot.maxStepPx = std::max(0.01, snapshot.maxSpeedPxPerSec * std::clamp(dtSec, 0.0005, 0.050));
    snapshot.calibratedCounts = config.target_calibrated_pixel_counts_enabled;
    snapshot.updatedAt = now;

    {
        std::lock_guard<std::mutex> lock(movementMtx);
        snapshot.carryX = movementCountCarryX;
        snapshot.carryY = movementCountCarryY;
    }

    if (state)
    {
        snapshot.hasState = state->valid;
        snapshot.sequence = state->sequence;
        snapshot.trackId = state->trackId;
        snapshot.observedThisFrame = state->observedThisFrame;
        snapshot.missedFrames = state->missedFrames;
        snapshot.confidence = state->confidence;
        snapshot.aimX = state->aimX;
        snapshot.aimY = state->aimY;
        if (state->observationTime.time_since_epoch().count() != 0)
        {
            snapshot.stateAgeMs = std::max(
                0.0,
                std::chrono::duration<double, std::milli>(now - state->observationTime).count());
        }
    }

    return snapshot;
}

void MouseThread::updateTargetStreamDebug(const TargetStreamDebugSnapshot& snapshot)
{
    if (!config.target_stream_debug_enabled && !targetSignalDiagnosticsActive())
        return;

    if (config.target_stream_debug_enabled)
    {
        std::lock_guard<std::mutex> lock(targetStreamDebugMutex);
        targetStreamDebug = snapshot;
    }

    if (targetSignalDiagnosticsActive())
        recordTargetSignalDiagnosticSample(snapshot, snapshot.updatedAt);
}

bool MouseThread::targetSignalDiagnosticsActive() const
{
    return config.target_signal_diagnostics_enabled;
}

void MouseThread::recordTargetSignalDiagnosticSample(
    const TargetStreamDebugSnapshot& snapshot,
    std::chrono::steady_clock::time_point now)
{
    if (!targetSignalDiagnosticsActive())
        return;

    if (now.time_since_epoch().count() == 0)
        now = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(targetSignalDiagnosticsMutex);
    if (targetSignalStartTime.time_since_epoch().count() == 0)
        targetSignalStartTime = now;

    const std::string status = snapshot.status ? snapshot.status : "";
    TargetSignalSample sample;
    sample.timeSec = std::chrono::duration<double>(now - targetSignalStartTime).count();
    sample.trackId = snapshot.trackId;
    sample.hasState = snapshot.hasState;
    sample.streaming = snapshot.streaming;
    sample.emittedMovement = snapshot.emittedMovement;
    sample.blocked = !snapshot.streaming &&
        status != "Inside deadzone" &&
        status != "Carry only" &&
        status != "Emitting" &&
        status != "No movement";
    sample.confidence = snapshot.confidence;
    sample.stateAgeMs = snapshot.stateAgeMs;
    sample.tickDtMs = snapshot.tickDtMs;
    sample.aimX = snapshot.aimX;
    sample.aimY = snapshot.aimY;
    sample.predictedX = snapshot.predictedX;
    sample.predictedY = snapshot.predictedY;
    sample.errorX = snapshot.errorX;
    sample.errorY = snapshot.errorY;
    sample.distancePx = snapshot.distancePx;
    sample.emittedPixelX = snapshot.emittedPixelX;
    sample.emittedPixelY = snapshot.emittedPixelY;
    sample.emittedCountRawX = snapshot.emittedCountRawX;
    sample.emittedCountRawY = snapshot.emittedCountRawY;
    sample.emittedCountX = snapshot.emittedCountX;
    sample.emittedCountY = snapshot.emittedCountY;
    sample.carryX = snapshot.carryX;
    sample.carryY = snapshot.carryY;
    sample.alpha = snapshot.alpha;
    sample.maxStepPx = snapshot.maxStepPx;
    sample.status = snapshot.status ? snapshot.status : "";

    targetSignalSamples.push_back(sample);
    const size_t maxSamples = static_cast<size_t>(std::clamp(config.target_signal_window_samples, 64, 2048));
    while (targetSignalSamples.size() > maxSamples)
        targetSignalSamples.pop_front();

    if (config.target_signal_logging_enabled)
    {
        const double intervalMs = std::clamp(static_cast<double>(config.target_signal_log_interval_ms), 1.0, 1000.0);
        if (targetSignalLastLogTime.time_since_epoch().count() == 0 ||
            std::chrono::duration<double, std::milli>(now - targetSignalLastLogTime).count() >= intervalMs)
        {
            appendTargetSignalLogRowLocked(sample);
            targetSignalLastLogTime = now;
        }
    }
    else if (targetSignalLogFile.is_open())
    {
        targetSignalLogFile.close();
        targetSignalLogPath.clear();
        targetSignalLogHeaderWritten = false;
    }

    targetSignalDiagnostics = computeTargetSignalDiagnosticsLocked(now);
}

void MouseThread::appendTargetSignalLogRowLocked(const TargetSignalSample& sample)
{
    std::string path = config.target_signal_log_file_path.empty()
        ? "logs/target_signal_diagnostics.csv"
        : config.target_signal_log_file_path;

    if (targetSignalLogPath != path || !targetSignalLogFile.is_open())
    {
        if (targetSignalLogFile.is_open())
            targetSignalLogFile.close();

        targetSignalLogPath = path;
        targetSignalLogHeaderWritten = false;

        const std::filesystem::path fsPath(path);
        const std::filesystem::path parent = fsPath.parent_path();
        if (!parent.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }

        std::error_code existsEc;
        const bool exists = std::filesystem::exists(fsPath, existsEc);
        std::error_code sizeEc;
        const auto existingSize = exists ? std::filesystem::file_size(fsPath, sizeEc) : 0;
        const bool needsHeader = !exists || sizeEc || existingSize == 0;
        targetSignalLogFile.open(path, std::ios::app);
        targetSignalLogHeaderWritten = !needsHeader;
    }

    if (!targetSignalLogFile.is_open())
        return;

    if (!targetSignalLogHeaderWritten)
    {
        targetSignalLogFile
            << "time_sec,track_id,status,has_state,streaming,emitted,blocked,confidence,state_age_ms,tick_dt_ms,"
            << "aim_x,aim_y,predicted_x,predicted_y,error_x,error_y,distance_px,"
            << "emitted_pixel_x,emitted_pixel_y,raw_count_x,raw_count_y,queued_count_x,queued_count_y,"
            << "carry_x,carry_y,alpha,max_step_px\n";
        targetSignalLogHeaderWritten = true;
    }

    targetSignalLogFile
        << std::fixed << std::setprecision(6)
        << sample.timeSec << ","
        << sample.trackId << ","
        << (sample.status ? sample.status : "") << ","
        << (sample.hasState ? 1 : 0) << ","
        << (sample.streaming ? 1 : 0) << ","
        << (sample.emittedMovement ? 1 : 0) << ","
        << (sample.blocked ? 1 : 0) << ","
        << sample.confidence << ","
        << sample.stateAgeMs << ","
        << sample.tickDtMs << ","
        << sample.aimX << ","
        << sample.aimY << ","
        << sample.predictedX << ","
        << sample.predictedY << ","
        << sample.errorX << ","
        << sample.errorY << ","
        << sample.distancePx << ","
        << sample.emittedPixelX << ","
        << sample.emittedPixelY << ","
        << sample.emittedCountRawX << ","
        << sample.emittedCountRawY << ","
        << sample.emittedCountX << ","
        << sample.emittedCountY << ","
        << sample.carryX << ","
        << sample.carryY << ","
        << sample.alpha << ","
        << sample.maxStepPx << "\n";
}

MouseThread::TargetSignalDiagnosticsSnapshot MouseThread::computeTargetSignalDiagnosticsLocked(
    std::chrono::steady_clock::time_point now) const
{
    TargetSignalDiagnosticsSnapshot out;
    out.enabled = config.target_signal_diagnostics_enabled;
    out.loggingEnabled = config.target_signal_logging_enabled;
    out.windowSamples = std::clamp(config.target_signal_window_samples, 64, 2048);
    out.logFilePath = config.target_signal_log_file_path.empty()
        ? "logs/target_signal_diagnostics.csv"
        : config.target_signal_log_file_path;
    out.updatedAt = now;

    const size_t n = targetSignalSamples.size();
    out.sampleCount = static_cast<int>(n);
    if (!out.enabled)
        return out;

    if (n < 2)
    {
        out.recommendation = "Collecting target signal samples.";
        return out;
    }

    const double windowSec = std::max(0.0, targetSignalSamples.back().timeSec - targetSignalSamples.front().timeSec);
    const double windowStartSec = targetSignalSamples.front().timeSec;
    out.windowSeconds = windowSec;

    double sumDt = 0.0;
    double sumDt2 = 0.0;
    double sumErr = 0.0;
    double sumErr2 = 0.0;
    double maxErr = 0.0;
    double sumEmit = 0.0;
    double sumEmit2 = 0.0;
    double emittedNetX = 0.0;
    double emittedNetY = 0.0;
    double sumSpeed = 0.0;
    double sumSpeed2 = 0.0;
    double sumSpeedDelta2 = 0.0;
    double lastSpeed = 0.0;
    double firstError = 0.0;
    bool haveLastSpeed = false;
    std::vector<double> outputSpeeds;
    outputSpeeds.reserve(n);
    double sumQueued = 0.0;
    int carryOnly = 0;
    int zeroOutput = 0;
    int blocked = 0;
    int speedDeltaCount = 0;
    int movingSpeedCount = 0;

    for (size_t sampleIndex = 0; sampleIndex < n; ++sampleIndex)
    {
        const auto& sample = targetSignalSamples[sampleIndex];
        const double dt = std::max(0.0, sample.tickDtMs);
        sumDt += dt;
        sumDt2 += dt * dt;

        const double err = std::hypot(sample.errorX, sample.errorY);
        if (sampleIndex == 0)
            firstError = err;
        sumErr += err;
        sumErr2 += err * err;
        maxErr = std::max(maxErr, err);

        const double emit = std::hypot(sample.emittedPixelX, sample.emittedPixelY);
        sumEmit += emit;
        sumEmit2 += emit * emit;
        out.outputPathLengthPx += emit;
        emittedNetX += sample.emittedPixelX;
        emittedNetY += sample.emittedPixelY;

        const double dtSec = dt * 0.001;
        const double speed = dtSec > 1e-6 ? emit / dtSec : 0.0;
        outputSpeeds.push_back(speed);
        if (speed > 1e-6)
        {
            ++movingSpeedCount;
            sumSpeed += speed;
            sumSpeed2 += speed * speed;
            if (speed > out.peakOutputSpeedPxPerSec)
            {
                out.peakOutputSpeedPxPerSec = speed;
                out.timeToPeakOutputMs = std::max(0.0, sample.timeSec - windowStartSec) * 1000.0;
            }
        }
        if (haveLastSpeed)
        {
            const double speedDelta = speed - lastSpeed;
            sumSpeedDelta2 += speedDelta * speedDelta;
            ++speedDeltaCount;
        }
        lastSpeed = speed;
        haveLastSpeed = true;

        const double queued = std::hypot(
            static_cast<double>(sample.emittedCountX),
            static_cast<double>(sample.emittedCountY));
        sumQueued += queued;

        if (sample.emittedMovement && sample.emittedCountX == 0 && sample.emittedCountY == 0)
            ++carryOnly;
        if (std::abs(sample.emittedPixelX) < 1e-9 && std::abs(sample.emittedPixelY) < 1e-9)
            ++zeroOutput;
        if (sample.blocked)
            ++blocked;
    }

    const double invN = 1.0 / static_cast<double>(n);
    out.avgDtMs = sumDt * invN;
    out.dtJitterMs = std::sqrt(std::max(0.0, sumDt2 * invN - out.avgDtMs * out.avgDtMs));
    out.sampleRateHz = out.avgDtMs > 1e-6 ? 1000.0 / out.avgDtMs : 0.0;
    out.avgErrorPx = sumErr * invN;
    out.rmsErrorPx = std::sqrt(sumErr2 * invN);
    out.peakErrorPx = maxErr;
    out.avgEmitPx = sumEmit * invN;
    out.rmsEmitPx = std::sqrt(sumEmit2 * invN);
    out.outputNetDisplacementPx = std::hypot(emittedNetX, emittedNetY);
    out.outputPathEfficiency = out.outputPathLengthPx > 1e-6
        ? std::clamp(out.outputNetDisplacementPx / out.outputPathLengthPx, 0.0, 1.0)
        : 1.0;
    out.endpointResidualPx = std::hypot(targetSignalSamples.back().errorX, targetSignalSamples.back().errorY);
    if (outputSpeeds.size() >= 3 && out.peakOutputSpeedPxPerSec > 1e-6)
    {
        const double peakThreshold = out.peakOutputSpeedPxPerSec * 0.15;
        for (size_t i = 1; i + 1 < outputSpeeds.size(); ++i)
        {
            if (outputSpeeds[i] > peakThreshold &&
                outputSpeeds[i] > outputSpeeds[i - 1] &&
                outputSpeeds[i] > outputSpeeds[i + 1])
            {
                ++out.outputSubmovementCount;
            }
        }
        if (out.outputPathLengthPx > 1.0)
            out.outputSubmovementCount = std::max(out.outputSubmovementCount, 1);
    }
    const double speedMean = movingSpeedCount > 0 ? sumSpeed / static_cast<double>(movingSpeedCount) : 0.0;
    const double speedVariance = movingSpeedCount > 0
        ? std::max(0.0, sumSpeed2 / static_cast<double>(movingSpeedCount) - speedMean * speedMean)
        : 0.0;
    const double speedStdDev = std::sqrt(speedVariance);
    const double deltaRms = speedDeltaCount > 0
        ? std::sqrt(sumSpeedDelta2 / static_cast<double>(speedDeltaCount))
        : 0.0;
    out.outputVelocityRoughness = out.peakOutputSpeedPxPerSec > 1e-6
        ? std::clamp((0.65 * speedStdDev + 0.35 * deltaRms) / out.peakOutputSpeedPxPerSec, 0.0, 5.0)
        : 0.0;
    const double fittsTargetWidth = std::max(8.0, static_cast<double>(config.target_deadzone_px) * 2.0);
    out.fittsIndexDifficulty = firstError > 1e-6
        ? std::log2(firstError / fittsTargetWidth + 1.0)
        : 0.0;
    out.fittsExpectedTimeMs = out.fittsIndexDifficulty > 0.0
        ? 50.0 + 150.0 * out.fittsIndexDifficulty
        : 0.0;
    out.fittsTimeRatio = out.fittsExpectedTimeMs > 1e-6
        ? (out.windowSeconds * 1000.0) / out.fittsExpectedTimeMs
        : 0.0;
    out.avgQueuedCounts = sumQueued * invN;
    out.queuedCountsPerSec = windowSec > 1e-6 ? sumQueued / windowSec : 0.0;
    out.carryOnlyRatio = static_cast<double>(carryOnly) * invN;
    out.zeroOutputRatio = static_cast<double>(zeroOutput) * invN;
    out.staleOrBlockedRatio = static_cast<double>(blocked) * invN;

    if (n >= 16 && out.sampleRateHz > 1.0)
    {
        std::vector<double> errorSignal;
        errorSignal.reserve(n);
        for (const auto& sample : targetSignalSamples)
            errorSignal.push_back(std::hypot(sample.errorX, sample.errorY));

        const double meanErr = out.avgErrorPx;
        const size_t maxBins = std::min<size_t>(n / 2, 96);
        constexpr double pi = 3.14159265358979323846;
        double bestMag = 0.0;
        double bestFreq = 0.0;
        for (size_t k = 1; k <= maxBins; ++k)
        {
            const double freq = static_cast<double>(k) * out.sampleRateHz / static_cast<double>(n);
            if (freq < 0.5 || freq > out.sampleRateHz * 0.45)
                continue;

            double re = 0.0;
            double im = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const double angle = -2.0 * pi * static_cast<double>(k) * static_cast<double>(i) / static_cast<double>(n);
                const double v = errorSignal[i] - meanErr;
                re += v * std::cos(angle);
                im += v * std::sin(angle);
            }

            const double mag = std::hypot(re, im) / static_cast<double>(n);
            if (mag > bestMag)
            {
                bestMag = mag;
                bestFreq = freq;
            }
        }
        out.dominantErrorFrequencyHz = bestFreq;
        out.dominantErrorMagnitude = bestMag;

        std::vector<double> emitSignal;
        emitSignal.reserve(n);
        double meanEmit = 0.0;
        for (const auto& sample : targetSignalSamples)
        {
            const double emit = std::hypot(sample.emittedPixelX, sample.emittedPixelY);
            emitSignal.push_back(emit);
            meanEmit += emit;
        }
        meanEmit *= invN;

        const int maxLag = static_cast<int>(std::min<size_t>(32, n / 4));
        double bestCorr = 0.0;
        int bestLag = 0;
        for (int lag = 0; lag <= maxLag; ++lag)
        {
            double xy = 0.0;
            double xx = 0.0;
            double yy = 0.0;
            int count = 0;
            for (size_t i = static_cast<size_t>(lag); i < n; ++i)
            {
                const double x = errorSignal[i - static_cast<size_t>(lag)] - meanErr;
                const double y = emitSignal[i] - meanEmit;
                xy += x * y;
                xx += x * x;
                yy += y * y;
                ++count;
            }

            if (count < 4 || xx <= 1e-12 || yy <= 1e-12)
                continue;

            const double corr = xy / std::sqrt(xx * yy);
            if (corr > bestCorr)
            {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        out.errorToOutputCorrelation = bestCorr;
        out.errorToOutputLagMs = static_cast<double>(bestLag) * out.avgDtMs;
        out.phaseLagDegrees = bestFreq > 1e-6
            ? std::fmod(360.0 * bestFreq * out.errorToOutputLagMs * 0.001, 360.0)
            : 0.0;
    }

    const double jitterRatio = out.avgDtMs > 1e-6 ? std::clamp(out.dtJitterMs / out.avgDtMs, 0.0, 1.0) : 1.0;
    out.cadenceHealth = 100.0 * (1.0 - jitterRatio);
    out.stabilityScore = std::clamp(
        100.0 -
        std::clamp(out.rmsErrorPx * 2.0, 0.0, 60.0) -
        std::clamp(out.staleOrBlockedRatio * 40.0, 0.0, 40.0) -
        std::clamp(out.carryOnlyRatio * 15.0, 0.0, 15.0),
        0.0,
        100.0);
    out.trajectoryQualityScore = std::clamp(
        100.0 -
        std::clamp((1.0 - out.outputPathEfficiency) * 45.0, 0.0, 45.0) -
        std::clamp(static_cast<double>(std::max(0, out.outputSubmovementCount - 2)) * 8.0, 0.0, 24.0) -
        std::clamp(out.outputVelocityRoughness * 30.0, 0.0, 30.0) -
        std::clamp(out.endpointResidualPx * 1.5, 0.0, 30.0),
        0.0,
        100.0);

    if (n < 64)
    {
        out.recommendation = "Collect more samples before tuning.";
    }
    else if (out.staleOrBlockedRatio > 0.25)
    {
        out.recommendation = "Autotune hint: target state is often blocked or stale; check confidence, state max age, and ID stability first.";
    }
    else if (out.avgDtMs < 0.9)
    {
        out.recommendation = "Autotune hint: stream cadence is faster than typical HID pacing; try target_stream_interval_ms near 1.0 ms.";
    }
    else if (jitterRatio > 0.35)
    {
        out.recommendation = "Autotune hint: stream cadence jitter is high; raise interval slightly or reduce CPU contention.";
    }
    else if (out.carryOnlyRatio > 0.60 && out.rmsErrorPx > config.target_deadzone_px + 1.0f)
    {
        out.recommendation = "Autotune hint: most movement is fractional carry; verify counts-per-pixel calibration or use a slightly slower stream interval.";
    }
    else if (out.outputPathLengthPx > 8.0 && out.outputPathEfficiency < 0.70)
    {
        out.recommendation = "Autotune hint: output path is inefficient; verify box/ID stability before increasing stream speed.";
    }
    else if (out.outputSubmovementCount > 4 && out.endpointResidualPx > config.target_deadzone_px + 2.0f)
    {
        out.recommendation = "Autotune hint: too many corrective submovements; reduce stream sharpness or prediction blend after checking tracker jitter.";
    }
    else if (out.outputVelocityRoughness > 0.85 && out.outputPathLengthPx > 8.0)
    {
        out.recommendation = "Autotune hint: velocity profile is rough; lower high-frequency correction or increase tracker smoothing.";
    }
    else if (out.fittsTimeRatio > 1.8 && out.endpointResidualPx > config.target_deadzone_px + 2.0f)
    {
        out.recommendation = "Autotune hint: correction is slower than expected for the current error; check counts-per-pixel calibration before raising sharpness.";
    }
    else if (out.dominantErrorFrequencyHz > 10.0 && out.dominantErrorMagnitude > 0.75)
    {
        out.recommendation = "Autotune hint: high-frequency aim noise detected; reduce prediction blend or increase tracker smoothing.";
    }
    else if (out.errorToOutputLagMs > 8.0 && out.errorToOutputCorrelation > 0.25)
    {
        out.recommendation = "Autotune hint: output trails target error; consider slightly higher prediction blend or stream sharpness.";
    }
    else
    {
        out.recommendation = "Signal looks usable; keep logging for backend-specific tuning baselines.";
    }

    return out;
}

void MouseThread::resetTargetSignalDiagnostics()
{
    std::lock_guard<std::mutex> lock(targetSignalDiagnosticsMutex);
    targetSignalSamples.clear();
    targetSignalDiagnostics = {};
    targetSignalStartTime = {};
    targetSignalLastLogTime = {};
    if (targetSignalLogFile.is_open())
        targetSignalLogFile.close();
    targetSignalLogPath.clear();
    targetSignalLogHeaderWritten = false;
}

void MouseThread::targetStreamWorkerLoop()
{
    try
    {
        auto lastTick = std::chrono::steady_clock::now();
        std::uint64_t activeSequence = 0;
        int activeTrackId = std::numeric_limits<int>::min();
        double appliedSinceObservationX = 0.0;
        double appliedSinceObservationY = 0.0;
        bool wasStreaming = false;

        while (!workerStop)
        {
            const auto now = std::chrono::steady_clock::now();
            double dtSec = std::chrono::duration<double>(now - lastTick).count();
            lastTick = now;
            dtSec = std::clamp(dtSec, 0.0005, 0.050);

            TargetMotionState state;
            const bool haveState = snapshotTargetMotionState(state);
            const bool streamEnabled = config.target_stream_enabled;
            const bool aimingActive = aiming.load(std::memory_order_relaxed);

            if (haveState && streamEnabled && aimingActive)
            {
                if (state.trackId != activeTrackId)
                {
                    resetDirectMovement();
                    activeTrackId = state.trackId;
                }

                if (state.sequence != activeSequence)
                {
                    activeSequence = state.sequence;
                    appliedSinceObservationX = 0.0;
                    appliedSinceObservationY = 0.0;
                }

                const bool streamed = dispatchTargetStreamMovement(
                    state,
                    now,
                    dtSec,
                    appliedSinceObservationX,
                    appliedSinceObservationY);

                if (!streamed)
                {
                    appliedSinceObservationX = 0.0;
                    appliedSinceObservationY = 0.0;
                    if (wasStreaming)
                    {
                        resetDirectMovement();
                        updateLastAppliedMouseDelta(0.0, 0.0);
                    }
                    wasStreaming = false;
                }
                else
                {
                    wasStreaming = true;
                }
            }
            else
            {
                const char* status = "No target state";
                if (!streamEnabled)
                    status = "Target stream disabled";
                else if (!aimingActive)
                    status = "Aim inactive";

                updateTargetStreamDebug(makeTargetStreamDebugSnapshot(
                    haveState ? &state : nullptr,
                    now,
                    dtSec,
                    status));

                if (wasStreaming)
                {
                    resetDirectMovement();
                    updateLastAppliedMouseDelta(0.0, 0.0);
                }
                wasStreaming = false;
                appliedSinceObservationX = 0.0;
                appliedSinceObservationY = 0.0;
                activeSequence = haveState ? state.sequence : 0;
                activeTrackId = haveState ? state.trackId : std::numeric_limits<int>::min();
            }

            const double intervalMs = std::clamp(
                static_cast<double>(config.target_stream_interval_ms),
                0.5,
                8.0);
            std::unique_lock<std::mutex> lock(targetStateMutex);
            const std::uint64_t waitSequence = targetStateSequence;
            targetStreamCv.wait_for(
                lock,
                std::chrono::duration<double, std::milli>(intervalMs),
                [&] {
                    return workerStop.load(std::memory_order_relaxed) ||
                        targetStateSequence != waitSequence;
                });
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Target stream worker crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Target stream worker crashed: unknown exception." << std::endl;
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

std::pair<double, double> MouseThread::predictStreamAimPoint(
    const TargetMotionState& state,
    double ageSec) const
{
    if (!std::isfinite(state.aimX) || !std::isfinite(state.aimY))
        return { state.aimX, state.aimY };

    ageSec = std::clamp(ageSec, 0.0, 0.250);
    double predictedX = state.aimX;
    double predictedY = state.aimY;
    if (std::isfinite(state.velocityX) && std::isfinite(state.velocityY))
    {
        predictedX += state.velocityX * ageSec;
        predictedY += state.velocityY * ageSec;
    }

    if (!std::isfinite(predictedX) || !std::isfinite(predictedY))
        return { state.aimX, state.aimY };

    double leadX = predictedX - state.aimX;
    double leadY = predictedY - state.aimY;
    const double maxLead = std::clamp(static_cast<double>(config.target_prediction_max_lead_px), 0.0, 40.0);
    if (maxLead <= 1e-6)
        return { state.aimX, state.aimY };

    const double leadMagnitude = std::hypot(leadX, leadY);
    if (leadMagnitude > maxLead && leadMagnitude > 1e-9)
    {
        const double scale = maxLead / leadMagnitude;
        leadX *= scale;
        leadY *= scale;
    }

    const double blend = std::clamp(static_cast<double>(config.target_prediction_blend), 0.0, 0.65);
    const double minConfidence = std::clamp(static_cast<double>(config.target_min_stream_confidence), 0.0, 0.95);
    const double confidenceRange = std::max(0.05, 1.0 - minConfidence);
    const double confidenceGate = std::clamp((state.confidence - minConfidence) / confidenceRange, 0.0, 1.0);
    const double effectiveBlend = blend * confidenceGate;

    return {
        state.aimX + leadX * effectiveBlend,
        state.aimY + leadY * effectiveBlend
    };
}

MouseThread::Move MouseThread::emitPixelMovement(
    double pixelDx,
    double pixelDy,
    std::chrono::steady_clock::time_point now)
{
    if (!std::isfinite(pixelDx) || !std::isfinite(pixelDy))
        return { 0, 0 };

    if (std::abs(pixelDx) < 1e-9 && std::abs(pixelDy) < 1e-9)
    {
        updateLastAppliedMouseDelta(0.0, 0.0);
        return { 0, 0 };
    }

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

    return { dx, dy };
}

bool MouseThread::dispatchTargetStreamMovement(
    const TargetMotionState& state,
    std::chrono::steady_clock::time_point now,
    double dtSec,
    double& appliedSinceObservationX,
    double& appliedSinceObservationY)
{
    if (!state.valid || state.observationTime.time_since_epoch().count() == 0)
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Invalid target state"));
        return false;
    }

    if (state.missedFrames > 2)
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Too many missed frames"));
        return false;
    }

    const double minConfidence = std::clamp(static_cast<double>(config.target_min_stream_confidence), 0.0, 0.95);
    if (!std::isfinite(state.confidence) || state.confidence < minConfidence)
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Low confidence"));
        return false;
    }

    double ageSec = std::chrono::duration<double>(now - state.observationTime).count();
    if (!std::isfinite(ageSec))
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Invalid state age"));
        return false;
    }
    ageSec = std::max(0.0, ageSec);

    const double maxAgeSec = std::clamp(
        static_cast<double>(config.target_state_max_age_ms),
        16.0,
        500.0) * 0.001;
    if (ageSec > maxAgeSec)
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Stale target state"));
        return false;
    }

    const auto aim = predictStreamAimPoint(state, ageSec);
    if (!std::isfinite(aim.first) || !std::isfinite(aim.second))
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Invalid predicted aim"));
        return false;
    }

    double pixelDx = aim.first - center_x - appliedSinceObservationX;
    double pixelDy = aim.second - center_y - appliedSinceObservationY;
    if (!std::isfinite(pixelDx) || !std::isfinite(pixelDy))
    {
        updateTargetStreamDebug(makeTargetStreamDebugSnapshot(&state, now, dtSec, "Invalid stream error"));
        return false;
    }

    TargetStreamDebugSnapshot debug = makeTargetStreamDebugSnapshot(&state, now, dtSec, "Evaluating");
    debug.streaming = true;
    debug.predictedX = aim.first;
    debug.predictedY = aim.second;
    debug.errorX = pixelDx;
    debug.errorY = pixelDy;
    debug.appliedSinceObservationX = appliedSinceObservationX;
    debug.appliedSinceObservationY = appliedSinceObservationY;

    const double distance = std::hypot(pixelDx, pixelDy);
    debug.distancePx = distance;
    const double deadzone = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
    debug.deadzonePx = deadzone;
    if (distance <= deadzone)
    {
        debug.status = "Inside deadzone";
        updateLastAppliedMouseDelta(0.0, 0.0);
        updateTargetStreamDebug(debug);
        return true;
    }

    const double sharpness = std::clamp(static_cast<double>(config.target_stream_sharpness), 1.0, 80.0);
    const double alpha = std::clamp(1.0 - std::exp(-sharpness * dtSec), 0.0, 1.0);
    debug.alpha = alpha;
    pixelDx *= alpha;
    pixelDy *= alpha;

    const double maxSpeed = std::clamp(static_cast<double>(config.target_max_pixel_speed), 50.0, 20000.0);
    const double maxStep = std::max(0.01, maxSpeed * dtSec);
    debug.maxSpeedPxPerSec = maxSpeed;
    debug.maxStepPx = maxStep;
    const double stepLength = std::hypot(pixelDx, pixelDy);
    if (stepLength > maxStep && stepLength > 1e-9)
    {
        const double scale = maxStep / stepLength;
        pixelDx *= scale;
        pixelDy *= scale;
    }

    if (std::abs(pixelDx) < 1e-9 && std::abs(pixelDy) < 1e-9)
    {
        debug.status = "No movement";
        updateTargetStreamDebug(debug);
        return true;
    }

    appliedSinceObservationX += pixelDx;
    appliedSinceObservationY += pixelDy;
    debug.appliedSinceObservationX = appliedSinceObservationX;
    debug.appliedSinceObservationY = appliedSinceObservationY;
    debug.emittedPixelX = pixelDx;
    debug.emittedPixelY = pixelDy;
    const auto rawCounts = pixelDeltaToCounts(pixelDx, pixelDy);
    debug.emittedCountRawX = rawCounts.first;
    debug.emittedCountRawY = rawCounts.second;
    const Move emitted = emitPixelMovement(pixelDx, pixelDy, now);
    debug.emittedCountX = emitted.dx;
    debug.emittedCountY = emitted.dy;
    debug.emittedMovement = true;
    debug.status = (emitted.dx != 0 || emitted.dy != 0) ? "Emitting" : "Carry only";
    {
        std::lock_guard<std::mutex> lock(movementMtx);
        debug.carryX = movementCountCarryX;
        debug.carryY = movementCountCarryY;
    }
    updateTargetStreamDebug(debug);
    return true;
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

    emitPixelMovement(pixelDx, pixelDy, std::chrono::steady_clock::now());
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

void MouseThread::publishTargetMotionState(const LockedTargetInfo& lockInfo)
{
    publishTargetMotionState(
        lockInfo.target,
        lockInfo.targetVelocityX,
        lockInfo.targetVelocityY,
        lockInfo.observedThisFrame,
        lockInfo.missedFrames);
}

void MouseThread::publishTargetMotionState(
    const BoxTarget& target,
    double velocityX,
    double velocityY,
    bool observedThisFrame,
    int missedFrames)
{
    if (!std::isfinite(target.smoothX) || !std::isfinite(target.smoothY))
    {
        clearTargetMotionState();
        return;
    }

    if (target.trackId != directMovementTrackId)
    {
        resetPrediction();
        directMovementTrackId = target.trackId;
    }

    const auto now = std::chrono::steady_clock::now();
    target_detected.store(true);
    last_target_time = now;

    const auto blendedAim = blendPredictedAimPoint(target.smoothX, target.smoothY, target.confidence);
    const double baseAimX = std::isfinite(blendedAim.first) ? blendedAim.first : target.smoothX;
    const double baseAimY = std::isfinite(blendedAim.second) ? blendedAim.second : target.smoothY;

    double streamVelocityX = std::isfinite(velocityX) ? velocityX : 0.0;
    double streamVelocityY = std::isfinite(velocityY) ? velocityY : 0.0;
    if (std::abs(streamVelocityX) < 1e-6 && std::isfinite(lastKalmanTelemetry.velocity_x))
        streamVelocityX = lastKalmanTelemetry.velocity_x;
    if (std::abs(streamVelocityY) < 1e-6 && std::isfinite(lastKalmanTelemetry.velocity_y))
        streamVelocityY = lastKalmanTelemetry.velocity_y;

    TargetMotionState next;
    next.valid = true;
    next.trackId = target.trackId;
    next.observedThisFrame = observedThisFrame;
    next.missedFrames = std::max(0, missedFrames);
    next.aimX = baseAimX;
    next.aimY = baseAimY;
    next.velocityX = streamVelocityX;
    next.velocityY = streamVelocityY;
    next.confidence = std::clamp(target.confidence, 0.0, 1.0);
    next.observationTime = now;
    next.publishTime = now;

    {
        std::lock_guard<std::mutex> lock(targetStateMutex);
        next.sequence = ++targetStateSequence;
        targetMotionState = next;
    }
    targetStreamCv.notify_all();
}

void MouseThread::clearTargetMotionState()
{
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(targetStateMutex);
        if (targetMotionState.valid)
        {
            targetMotionState = TargetMotionState{};
            targetMotionState.sequence = ++targetStateSequence;
            changed = true;
        }
    }

    if (changed)
        targetStreamCv.notify_all();
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
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        std::queue<Move> empty;
        moveQueue.swap(empty);
    }

    clearTargetMotionState();
    resetWindState();
    resetEgoMotionCompensation();
    resetDirectMovement();
    updateLastAppliedMouseDelta(0.0, 0.0);
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
    clearTargetMotionState();
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

void MouseThread::clearTargetSignalDiagnostics()
{
    resetTargetSignalDiagnostics();
}

MouseThread::TargetStreamDebugSnapshot MouseThread::getTargetStreamDebugSnapshot() const
{
    TargetStreamDebugSnapshot snapshot;
    snapshot.enabled = config.target_stream_debug_enabled;
    snapshot.streamEnabled = config.target_stream_enabled;
    snapshot.aimingActive = aiming.load(std::memory_order_relaxed);
    snapshot.centerX = center_x;
    snapshot.centerY = center_y;
    snapshot.deadzonePx = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
    snapshot.maxSpeedPxPerSec = std::clamp(static_cast<double>(config.target_max_pixel_speed), 50.0, 20000.0);
    snapshot.calibratedCounts = config.target_calibrated_pixel_counts_enabled;

    if (!snapshot.enabled)
        return snapshot;

    {
        std::lock_guard<std::mutex> lock(targetStreamDebugMutex);
        snapshot = targetStreamDebug;
    }

    snapshot.enabled = true;
    snapshot.streamEnabled = config.target_stream_enabled;
    snapshot.aimingActive = aiming.load(std::memory_order_relaxed);
    if (snapshot.updatedAt.time_since_epoch().count() == 0)
    {
        snapshot.status = "Waiting for stream tick";
        snapshot.centerX = center_x;
        snapshot.centerY = center_y;
        snapshot.deadzonePx = std::clamp(static_cast<double>(config.target_deadzone_px), 0.0, 20.0);
        snapshot.maxSpeedPxPerSec = std::clamp(static_cast<double>(config.target_max_pixel_speed), 50.0, 20000.0);
        snapshot.calibratedCounts = config.target_calibrated_pixel_counts_enabled;
        return snapshot;
    }

    snapshot.snapshotAgeMs = std::max(
        0.0,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - snapshot.updatedAt).count());
    return snapshot;
}

MouseThread::TargetSignalDiagnosticsSnapshot MouseThread::getTargetSignalDiagnosticsSnapshot() const
{
    TargetSignalDiagnosticsSnapshot snapshot;
    snapshot.enabled = config.target_signal_diagnostics_enabled;
    snapshot.loggingEnabled = config.target_signal_logging_enabled;
    snapshot.windowSamples = std::clamp(config.target_signal_window_samples, 64, 2048);
    snapshot.logFilePath = config.target_signal_log_file_path.empty()
        ? "logs/target_signal_diagnostics.csv"
        : config.target_signal_log_file_path;

    if (!snapshot.enabled)
        return snapshot;

    std::lock_guard<std::mutex> lock(targetSignalDiagnosticsMutex);
    snapshot = computeTargetSignalDiagnosticsLocked(std::chrono::steady_clock::now());
    if (snapshot.sampleCount == 0)
        snapshot.recommendation = "Waiting for target signal samples.";
    return snapshot;
}

void MouseThread::setMouseInput(IMouseInput* newMouseInput)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    mouseInput = newMouseInput;
}
