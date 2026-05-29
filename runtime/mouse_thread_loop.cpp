#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <chrono>
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

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<float> confidences;
    MultiTargetTracker targetTracker;
    std::optional<BoxTarget> activeTarget;
    std::pair<double, double> learnedPredictionLead{ 0.0, 0.0 };
    auto lastTrackerUpdate = std::chrono::steady_clock::time_point::min();
    unsigned long long seenDetectionResolutionGeneration =
        detection_resolution_generation.load(std::memory_order_relaxed);

    while (!shouldExit)
    {
        bool hasNewDetection = false;
        bool hasAimObservation = false;
        std::chrono::steady_clock::time_point bufferCaptureTimestamp{};
        std::chrono::steady_clock::time_point bufferPublishTimestamp{};

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
                    config.fovX,
                    config.fovY,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            targetTracker.reset();
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
                mouseThread.clearQueuedMoves();
                hasNewDetection = false;
            }
        }

        if (hasNewDetection)
        {
            targetTracker.update(
                boxes,
                classes,
                confidences,
                config.detection_resolution,
                config.detection_resolution,
                config.disable_headshot,
                aiming.load()
            );
            lastTrackerUpdate =
                (bufferCaptureTimestamp.time_since_epoch().count() != 0)
                ? bufferCaptureTimestamp
                : bufferPublishTimestamp;
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks = targetTracker.getDebugTracks();
                g_trackerLockedId = targetTracker.getLockedTrackId();
            }

            LockedTargetInfo lockInfo;
            if (targetTracker.getLockedTarget(lockInfo) && (lockInfo.observedThisFrame || lockInfo.missedFrames <= 2))
            {
                activeTarget = lockInfo.target;
                learnedPredictionLead = { 0.0, 0.0 };
                hasAimObservation = true;
                mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
                mouseThread.setTargetDetected(true);

                if (config.temporal_prediction_enabled && !lockInfo.predictedFuture.empty())
                {
                    mouseThread.storeFuturePositions(lockInfo.predictedFuture);
                    learnedPredictionLead = mouseThread.computePredictionFeedForwardLead(*activeTarget, lockInfo);
                }
                else
                {
                    auto futurePositions = mouseThread.predictFuturePositions(
                        activeTarget->smoothX,
                        activeTarget->smoothY,
                        config.prediction_futurePositions
                    );
                    mouseThread.storeFuturePositions(futurePositions);
                }
            }
            else
            {
                activeTarget.reset();
                learnedPredictionLead = { 0.0, 0.0 };
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearQueuedMoves();
            }
        }

        if (activeTarget)
        {
            const int fps = std::max(1, captureFps.load());
            const int staleMs = std::clamp(2000 / fps, 25, 180);
            if (std::chrono::steady_clock::now() - lastTrackerUpdate > std::chrono::milliseconds(staleMs))
            {
                activeTarget.reset();
                learnedPredictionLead = { 0.0, 0.0 };
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearQueuedMoves();
            }
        }

        if (aiming)
        {
            if (activeTarget && hasAimObservation)
            {
                mouseThread.moveMousePivot(
                    activeTarget->smoothX,
                    activeTarget->smoothY,
                    activeTarget->w,
                    activeTarget->h,
                    activeTarget->confidence,
                    0.0, 0.0,
                    learnedPredictionLead.first,
                    learnedPredictionLead.second);

                if (config.auto_shoot)
                {
                    mouseThread.pressMouse(*activeTarget);
                }
            }
            else
            {
                if (!activeTarget)
                {
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
            mouseThread.clearQueuedMoves();
            if (config.auto_shoot)
            {
                mouseThread.releaseMouse();
            }
        }

        mouseThread.checkAndResetPredictions();
    }
}


