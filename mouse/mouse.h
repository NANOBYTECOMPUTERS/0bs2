#ifndef MOUSE_H
#define MOUSE_H

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <utility>
#include <queue>
#include <thread>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <random>

#include "BoxTarget.h"
#include "MouseInput.h"
#include "aim_kalman.h"
#include "ego_motion_compensator.h"

class MouseThread
{
public:
    struct TargetStreamDebugSnapshot
    {
        bool enabled = false;
        bool streamEnabled = false;
        bool aimingActive = false;
        bool hasState = false;
        bool streaming = false;
        bool emittedMovement = false;
        bool calibratedCounts = false;
        const char* status = "Debug disabled";
        std::uint64_t sequence = 0;
        int trackId = -1;
        bool observedThisFrame = false;
        int missedFrames = 0;
        double confidence = 0.0;
        double stateAgeMs = 0.0;
        double snapshotAgeMs = 0.0;
        double tickDtMs = 0.0;
        double centerX = 0.0;
        double centerY = 0.0;
        double aimX = 0.0;
        double aimY = 0.0;
        double predictedX = 0.0;
        double predictedY = 0.0;
        double errorX = 0.0;
        double errorY = 0.0;
        double appliedSinceObservationX = 0.0;
        double appliedSinceObservationY = 0.0;
        double emittedPixelX = 0.0;
        double emittedPixelY = 0.0;
        double emittedCountRawX = 0.0;
        double emittedCountRawY = 0.0;
        int emittedCountX = 0;
        int emittedCountY = 0;
        double carryX = 0.0;
        double carryY = 0.0;
        double distancePx = 0.0;
        double deadzonePx = 0.0;
        double alpha = 0.0;
        double maxStepPx = 0.0;
        double maxSpeedPxPerSec = 0.0;
        std::chrono::steady_clock::time_point updatedAt{};
    };

private:
    double screen_width;
    double screen_height;
    double center_x;
    double center_y;
    bool   auto_shoot;
    float  bScope_multiplier;

    double prev_x, prev_y;
    double prev_velocity_x, prev_velocity_y;
    std::chrono::time_point<std::chrono::steady_clock> prev_time;
    std::chrono::steady_clock::time_point last_target_time;
    std::atomic<bool> target_detected{ false };
    std::atomic<bool> mouse_pressed{ false };

    IMouseInput* mouseInput;

    void sendMovementToDriver(int dx, int dy);

    struct Move { int dx; int dy; };

    struct TargetMotionState
    {
        bool valid = false;
        int trackId = -1;
        bool observedThisFrame = false;
        int missedFrames = 0;
        double aimX = 0.0;
        double aimY = 0.0;
        double velocityX = 0.0;
        double velocityY = 0.0;
        double confidence = 0.0;
        std::chrono::steady_clock::time_point observationTime{};
        std::chrono::steady_clock::time_point publishTime{};
        std::uint64_t sequence = 0;
    };

    std::queue<Move>              moveQueue;
    std::mutex                    queueMtx;
    std::condition_variable       queueCv;
    const size_t                  queueLimit = 5;
    std::thread                   moveWorker;
    std::thread                   targetStreamWorker;
    std::atomic<bool>             workerStop{ false };

    mutable std::mutex            targetStateMutex;
    std::condition_variable       targetStreamCv;
    TargetMotionState             targetMotionState;
    std::uint64_t                 targetStateSequence = 0;
    mutable std::mutex            targetStreamDebugMutex;
    TargetStreamDebugSnapshot     targetStreamDebug;

    std::mutex                    movementMtx;
    double                        movementCountCarryX = 0.0;
    double                        movementCountCarryY = 0.0;
    mutable std::mutex            lastAppliedMouseDeltaMutex;
    std::pair<double, double>     lastAppliedMouseDelta{ 0.0, 0.0 };

    std::vector<std::pair<double, double>> futurePositions;
    std::mutex                    futurePositionsMutex;
    aim::AimKalman2D              targetKalman;
    aim::AimKalmanTelemetry       lastKalmanTelemetry;
    double                        lastPredictionLookaheadSec = 0.0;
    int                           directMovementTrackId = -1;
    aim::EgoMotionCompensator     egoMotionCompensator;
    mutable std::mutex            egoMotionVelocityMutex;
    std::chrono::steady_clock::time_point egoMotionVelocityLastTimestamp{};
    double                        egoMotionCameraVelocityPxPerSec = 0.0;

    void moveWorkerLoop();
    void targetStreamWorkerLoop();
    void queueMove(int dx, int dy);
    bool snapshotTargetMotionState(TargetMotionState& out) const;
    void updateTargetStreamDebug(const TargetStreamDebugSnapshot& snapshot);
    TargetStreamDebugSnapshot makeTargetStreamDebugSnapshot(
        const TargetMotionState* state,
        std::chrono::steady_clock::time_point now,
        double dtSec,
        const char* status);
    Move emitPixelMovement(
        double pixelDx,
        double pixelDy,
        std::chrono::steady_clock::time_point now);
    bool dispatchTargetStreamMovement(
        const TargetMotionState& state,
        std::chrono::steady_clock::time_point now,
        double dtSec,
        double& appliedSinceObservationX,
        double& appliedSinceObservationY);
    std::pair<double, double> predictStreamAimPoint(
        const TargetMotionState& state,
        double ageSec) const;
    void dispatchTargetMovement(
        double pivotX,
        double pivotY,
        double targetOffsetX = 0.0,
        double targetOffsetY = 0.0);
    void updateLastAppliedMouseDelta(double dx, double dy);
    void recordEgoMotionDelta(double pixelDx, double pixelDy, std::chrono::steady_clock::time_point timestamp);
    void updateEgoMotionVelocityEstimate(double pixelDx, double pixelDy, std::chrono::steady_clock::time_point timestamp);
    double currentEgoMotionCameraVelocityPxPerSec() const;
    std::pair<double, double> pixelDeltaToCounts(double pixelDx, double pixelDy) const;
    void resetDirectMovement();

    void   resetWindState();
    void   appendWindDebugStep(int dx, int dy);
    void   pruneWindDebugTrailLocked(const std::chrono::steady_clock::time_point& now);

    struct WindDebugPoint
    {
        double x = 0.0;
        double y = 0.0;
        std::chrono::steady_clock::time_point t{};
    };

    // Persistent wind state to avoid per-frame "reset" feel.
    double windCarryX = 0.0;
    double windCarryY = 0.0;
    double windVelX = 0.0;
    double windVelY = 0.0;
    double windNoiseX = 0.0;
    double windNoiseY = 0.0;
    double windFracX = 0.0;
    double windFracY = 0.0;
    double windPatternX = 0.0;
    double windPatternY = 0.0;
    double windPatternPhaseA = 0.0;
    double windPatternPhaseB = 0.0;
    double windPatternRateA = 0.0;
    double windPatternRateB = 0.0;
    std::mt19937 windRng{ std::random_device{}() };

    std::deque<WindDebugPoint> windDebugTrail;
    std::mutex                             windDebugTrailMutex;
    double                                 windDebugCursorX = 0.0;
    double                                 windDebugCursorY = 0.0;

    double currentDetectionDelaySec() const;
    double currentPredictionLookaheadSec(double detectionDelaySec) const;
    std::pair<double, double> blendPredictedAimPoint(double pivotX, double pivotY, double confidence);

public:
    std::mutex input_method_mutex;

    MouseThread(
        int  resolution,
        bool auto_shoot,
        float bScope_multiplier,
        IMouseInput* mouseInputDevice = nullptr
    );
    ~MouseThread();

    void updateConfig(
        int resolution,
        bool auto_shoot,
        float bScope_multiplier
    );
    void updateDetectionGeometry(int width, int height);

    void moveMousePivot(double pivotX, double pivotY);
    void moveMousePivot(double pivotX, double pivotY, double targetWidth, double targetHeight, double confidence = 1.0);
    void moveMousePivot(
        double pivotX,
        double pivotY,
        double targetWidth,
        double targetHeight,
        double confidence,
        double targetOffsetX,
        double targetOffsetY);
    void moveMouseTarget(const BoxTarget& target);
    void publishTargetMotionState(const LockedTargetInfo& lockInfo);
    void publishTargetMotionState(
        const BoxTarget& target,
        double velocityX = 0.0,
        double velocityY = 0.0,
        bool observedThisFrame = true,
        int missedFrames = 0);
    void clearTargetMotionState();
    void clearQueuedMoves();
    std::pair<double, double> consumeEgoMotionCompensation(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end);
    void resetEgoMotionCompensation();
    std::pair<double, double> getLastAppliedMouseDelta() const;
    std::pair<double, double> predict_target_position(double target_x, double target_y);
    void moveMouse(const BoxTarget& target);
    void pressMouse(const BoxTarget& target);
    void releaseMouse();
    void resetPrediction();
    void checkAndResetPredictions();
    bool check_target_in_scope(double target_x, double target_y,
        double target_w, double target_h, double reduction_factor);

    std::vector<std::pair<double, double>> predictFuturePositions(double pivotX, double pivotY, int frames);
    void storeFuturePositions(const std::vector<std::pair<double, double>>& positions);
    void clearFuturePositions();
    std::vector<std::pair<double, double>> getFuturePositions();
    void clearWindDebugTrail();
    std::vector<std::pair<double, double>> getWindDebugTrail();
    TargetStreamDebugSnapshot getTargetStreamDebugSnapshot() const;

    void moveRelative(int dx, int dy);
    void setMouseInput(IMouseInput* newMouseInput);

    void setTargetDetected(bool detected);
    void setLastTargetTime(const std::chrono::steady_clock::time_point& t) { last_target_time = t; }
};

#endif // MOUSE_H
