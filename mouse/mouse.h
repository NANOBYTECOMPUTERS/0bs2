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
#include <deque>
#include <random>

#include "BoxTarget.h"
#include "MouseInput.h"
#include "aim_kalman.h"
#include "PidMouseController.h"

class MouseThread
{
private:
    double screen_width;
    double screen_height;
    double fov_x;
    double fov_y;
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

    std::queue<Move>              moveQueue;
    std::mutex                    queueMtx;
    std::condition_variable       queueCv;
    const size_t                  queueLimit = 5;
    std::thread                   moveWorker;
    std::atomic<bool>             workerStop{ false };

    aim::PidMouseController       pidController;
    std::thread                   pidActuator;
    std::atomic<bool>             pidStop{ false };
    std::mutex                    pidMtx;
    double                        pidCountCarryX = 0.0;
    double                        pidCountCarryY = 0.0;

    std::vector<std::pair<double, double>> futurePositions;
    std::mutex                    futurePositionsMutex;
    aim::AimKalman2D              targetKalman;
    aim::AimKalmanTelemetry       lastKalmanTelemetry;
    double                        lastPredictionLookaheadSec = 0.0;

    void moveWorkerLoop();
    void queueMove(int dx, int dy);
    void pidActuatorLoop();
    void publishPidObservation(
        double pivotX,
        double pivotY,
        double targetWidth,
        double targetHeight,
        double confidence,
        double targetOffsetX = 0.0,
        double targetOffsetY = 0.0);
    std::pair<double, double> pixelDeltaToCounts(double pixelDx, double pixelDy) const;
    void resetPid();

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

public:
    std::mutex input_method_mutex;

    MouseThread(
        int  resolution,
        int  fovX,
        int  fovY,
        bool auto_shoot,
        float bScope_multiplier,
        IMouseInput* mouseInputDevice = nullptr
    );
    ~MouseThread();

    void updateConfig(
        int resolution,
        int fovX,
        int fovY,
        bool auto_shoot,
        float bScope_multiplier
    );

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
    void clearQueuedMoves();
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

    void moveRelative(int dx, int dy);
    void setMouseInput(IMouseInput* newMouseInput);

    void setTargetDetected(bool detected);
    void setLastTargetTime(const std::chrono::steady_clock::time_point& t) { last_target_time = t; }
};

#endif // MOUSE_H
