#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "capture.h"
#include "mouse.h"
#include "0BS_box_2.h"
#include "runtime/thread_loops.h"

void createInputDevices();
void assignInputDevices();
void handleEasyNoRecoil(MouseThread& mouseThread)
{
    if (config.easynorecoil && shooting.load() && zooming.load())
    {
        int recoil_compensation = static_cast<int>(config.easynorecoilstrength);
        mouseThread.moveRelative(0, recoil_compensation);
    }
}

namespace
{
std::optional<BoxTarget> chooseDirectDetectionTarget(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    const std::vector<float>& confidences,
    int frameWidth,
    int frameHeight)
{
    if (boxes.empty() || boxes.size() != classes.size())
        return std::nullopt;

    const double centerX = std::max(1, frameWidth) * 0.5;
    const double centerY = std::max(1, frameHeight) * 0.5;
    const double confidenceBiasScale = std::max(1, std::max(frameWidth, frameHeight)) * 0.04;
    int bestIndex = -1;
    double bestScore = std::numeric_limits<double>::infinity();
    double bestPivotX = 0.0;
    double bestPivotY = 0.0;
    double bestConfidence = 1.0;

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const int cls = classes[i];
        if (config.disable_headshot)
        {
            if (cls != config.class_player)
                continue;
        }
        else if (cls != config.class_player && cls != config.class_head)
        {
            continue;
        }

        const cv::Rect& box = boxes[i];
        if (box.width <= 0 || box.height <= 0)
            continue;

        const double confidence = (i < confidences.size())
            ? std::clamp(static_cast<double>(confidences[i]), 0.0, 1.0)
            : 1.0;

        double yBias = 0.5;
        if (cls == config.class_head)
        {
            yBias = std::clamp(
                static_cast<double>(config.head_y_offset),
                static_cast<double>(Config::kHeadYOffsetMin),
                static_cast<double>(Config::kHeadYOffsetMax));
        }
        else if (cls == config.class_player)
        {
            yBias = std::clamp(
                static_cast<double>(config.body_y_offset),
                static_cast<double>(Config::kBodyYOffsetMin),
                static_cast<double>(Config::kBodyYOffsetMax));
        }

        const double pivotX = static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5;
        const double pivotY = static_cast<double>(box.y) + static_cast<double>(box.height) * yBias;
        const double distance = std::hypot(pivotX - centerX, pivotY - centerY);
        const double score = distance - confidence * confidenceBiasScale;
        if (score < bestScore)
        {
            bestScore = score;
            bestIndex = static_cast<int>(i);
            bestPivotX = pivotX;
            bestPivotY = pivotY;
            bestConfidence = confidence;
        }
    }

    if (bestIndex < 0)
        return std::nullopt;

    const cv::Rect& box = boxes[static_cast<size_t>(bestIndex)];
    BoxTarget target(
        box.x,
        box.y,
        box.width,
        box.height,
        classes[static_cast<size_t>(bestIndex)],
        bestPivotX,
        bestPivotY,
        bestConfidence,
        -1);
    target.smoothX = bestPivotX;
    target.smoothY = bestPivotY;
    return target;
}
}

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<float> confidences;
    MultiTargetTracker targetTracker;
    std::optional<BoxTarget> activeTarget;
    auto lastTrackerUpdate = std::chrono::steady_clock::time_point::min();
    unsigned long long seenDetectionResolutionGeneration =
        detection_resolution_generation.load(std::memory_order_relaxed);

    while (!shouldExit)
    {
        bool hasNewDetection = false;
        bool hasAimObservation = false;
        std::chrono::steady_clock::time_point bufferCaptureTimestamp{};
        std::chrono::steady_clock::time_point bufferPublishTimestamp{};
        CaptureFrameGeometry bufferGeometry;

        {
            std::unique_lock<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return detectionBuffer.version > lastVersion || shouldExit;
                }
            );

            if (shouldExit) break;

            if (detectionBuffer.version > lastVersion)
            {
                boxes = detectionBuffer.boxes;
                classes = detectionBuffer.classes;
                confidences = detectionBuffer.confidences;
                bufferCaptureTimestamp = detectionBuffer.captureTimestamp;
                bufferPublishTimestamp = detectionBuffer.publishTimestamp;
                bufferGeometry = detectionBuffer.geometry;
                lastVersion = detectionBuffer.version;
                hasNewDetection = true;
            }
        }

        if (input_method_changed.load())
        {
            createInputDevices();
            assignInputDevices();
            input_method_changed.store(false);
        }

        const unsigned long long currentDetectionResolutionGeneration =
            detection_resolution_generation.load(std::memory_order_relaxed);
        if (currentDetectionResolutionGeneration != seenDetectionResolutionGeneration)
        {
            seenDetectionResolutionGeneration = currentDetectionResolutionGeneration;
            {
                std::lock_guard<std::recursive_mutex> cfgLock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            targetTracker.reset();
            mouseThread.resetEgoMotionCompensation();
            mouseThread.clearTargetMotionState();
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks.clear();
                g_trackerLockedId = -1;
            }
        }

        if (hasNewDetection)
        {
            const int fps = std::max(1, captureFps.load());
            const int maxDetectionAgeMs = std::clamp(2000 / fps, 25, 180);
            const auto now = std::chrono::steady_clock::now();
            if (bufferCaptureTimestamp.time_since_epoch().count() != 0 &&
                now - bufferCaptureTimestamp > std::chrono::milliseconds(maxDetectionAgeMs))
            {
                activeTarget.reset();
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearTargetMotionState();
                mouseThread.clearQueuedMoves();
                hasNewDetection = false;
            }
        }

        if (hasNewDetection)
        {
            const auto trackerFrameTimestamp =
                (bufferCaptureTimestamp.time_since_epoch().count() != 0)
                ? bufferCaptureTimestamp
                : ((bufferPublishTimestamp.time_since_epoch().count() != 0)
                    ? bufferPublishTimestamp
                    : std::chrono::steady_clock::now());
            std::pair<double, double> egoMotionShift{ 0.0, 0.0 };
            if (lastTrackerUpdate.time_since_epoch().count() != 0)
            {
                egoMotionShift = mouseThread.consumeEgoMotionCompensation(
                    lastTrackerUpdate,
                    trackerFrameTimestamp);
            }
            else
            {
                mouseThread.resetEgoMotionCompensation();
            }

            const int trackerFrameWidth = bufferGeometry.hasValidModel()
                ? bufferGeometry.modelWidth
                : config.detection_resolution;
            const int trackerFrameHeight = bufferGeometry.hasValidModel()
                ? bufferGeometry.modelHeight
                : config.detection_resolution;

            mouseThread.updateDetectionGeometry(trackerFrameWidth, trackerFrameHeight);

            targetTracker.update(
                boxes,
                classes,
                confidences,
                trackerFrameWidth,
                trackerFrameHeight,
                config.disable_headshot,
                aiming.load(),
                cv::Point2d(egoMotionShift.first, egoMotionShift.second)
            );
            lastTrackerUpdate = trackerFrameTimestamp;
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks = targetTracker.getDebugTracks();
                g_trackerLockedId = targetTracker.getLockedTrackId();
            }

            LockedTargetInfo lockInfo;
            if (targetTracker.getLockedTarget(lockInfo) && (lockInfo.observedThisFrame || lockInfo.missedFrames <= 2))
            {
                activeTarget = lockInfo.target;
                hasAimObservation = true;
                mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
                mouseThread.setTargetDetected(true);
                mouseThread.publishTargetMotionState(lockInfo);

                auto futurePositions = mouseThread.predictFuturePositions(
                    activeTarget->smoothX,
                    activeTarget->smoothY,
                    config.prediction_futurePositions
                );
                mouseThread.storeFuturePositions(futurePositions);
            }
            else
            {
                auto directTarget = chooseDirectDetectionTarget(
                    boxes,
                    classes,
                    confidences,
                    trackerFrameWidth,
                    trackerFrameHeight);
                if (directTarget)
                {
                    activeTarget = *directTarget;
                    hasAimObservation = true;
                    mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
                    mouseThread.setTargetDetected(true);
                    mouseThread.publishTargetMotionState(*directTarget);
                    auto futurePositions = mouseThread.predictFuturePositions(
                        activeTarget->smoothX,
                        activeTarget->smoothY,
                        config.prediction_futurePositions
                    );
                    mouseThread.storeFuturePositions(futurePositions);
                }
                else
                {
                    activeTarget.reset();
                    mouseThread.clearFuturePositions();
                    mouseThread.setTargetDetected(false);
                    mouseThread.clearTargetMotionState();
                    mouseThread.clearQueuedMoves();
                }
            }
        }

        if (activeTarget)
        {
            const int fps = std::max(1, captureFps.load());
            const int staleMs = std::clamp(2000 / fps, 25, 180);
            if (std::chrono::steady_clock::now() - lastTrackerUpdate > std::chrono::milliseconds(staleMs))
            {
                activeTarget.reset();
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearTargetMotionState();
                mouseThread.clearQueuedMoves();
            }
        }

        if (aiming)
        {
            if (activeTarget && hasAimObservation)
            {
                if (config.auto_shoot)
                {
                    mouseThread.pressMouse(*activeTarget);
                }
            }
            else
            {
                if (!activeTarget)
                {
                    mouseThread.clearTargetMotionState();
                    mouseThread.clearQueuedMoves();
                }

                if (config.auto_shoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }
        else
        {
            mouseThread.clearTargetMotionState();
            mouseThread.clearQueuedMoves();
            if (config.auto_shoot)
            {
                mouseThread.releaseMouse();
            }
        }

        mouseThread.checkAndResetPredictions();
    }
}


