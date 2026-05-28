#ifndef BOXTARGET_H
#define BOXTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <chrono>
#include <random>
#include <utility>

#include "aim_kalman.h"
#include "neural/TemporalPredictor.h"

class BoxTarget
{
public:
    BoxTarget();
    int x, y, w, h;
    int classId;

    double pivotX;
    double pivotY;
    double smoothX;
    double smoothY;
    double confidence;
    int trackId;

    BoxTarget(
        int x,
        int y,
        int w,
        int h,
        int classId,
        double pivotX = 0.0,
        double pivotY = 0.0,
        double confidence = 1.0,
        int trackId = -1);
};

BoxTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot,
    const std::vector<float>* confidences = nullptr
    );

struct InnerAimTrack {
    int trackId = -1;
    int classId = -1;

    cv::Rect outerBox;

    double smoothX = 0.0;
    double smoothY = 0.0;
    float  radius = 8.0f;

    float confidence = 0.0f;
    float consistencyScore = 0.0f;

    int hits = 0;
    int missedFrames = 0;
    bool observedThisFrame = false;

    aim::AimKalman2D kalman;
};

struct LockedTargetInfo
{
    int trackId = -1;
    bool observedThisFrame = false;
    int missedFrames = 0;
    BoxTarget target;
    std::vector<std::pair<double, double>> predictedFuture;
    int predictedFutureAgeFrames = 9999;
    double targetVelocityX = 0.0;
    double targetVelocityY = 0.0;
    double targetBoxScaleVelocity = 0.0;
};

struct TrackDebugInfo
{
    int trackId = -1;
    int classId = -1;
    cv::Rect box;
    double pivotX = 0.0;
    double pivotY = 0.0;
    double innerAimX = 0.0;
    double innerAimY = 0.0;
    float innerAimRadius = 0.0f;
    float consistencyScore = 0.0f;
    float confidence = 1.0f;
    int hits = 0;
    bool observedThisFrame = false;
    int missedFrames = 0;
    bool isLocked = false;
    double lastAssociationScore = 0.0;
    double lastAssociationDistancePx = 0.0;
    double lastAssociationIou = 0.0;
    double lastHeadingAlignment = 0.0;
    double lastNeuralScore = 0.5;
    double lastNeuralBonus = 0.0;
    bool lastNeuralEvaluated = false;
    std::vector<std::pair<double, double>> temporalFuture;
    bool temporalPredictionValid = false;
};

class MultiTargetTracker
{
public:
    void reset();
    void update(
        const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classes,
        const std::vector<float>& confidences,
        int screenWidth,
        int screenHeight,
        bool disableHeadshot,
        bool keepCurrentLock
    );
    void update(
        const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classes,
        int screenWidth,
        int screenHeight,
        bool disableHeadshot,
        bool keepCurrentLock
    );
    bool getLockedTarget(LockedTargetInfo& out) const;
    int getLockedTrackId() const { return lockedTrackId_; }
    std::vector<TrackDebugInfo> getDebugTracks() const;

private:
    struct TrackState
    {
        struct TrackHistorySample
        {
            double x = 0.0;
            double y = 0.0;
            double w = 0.0;
            double h = 0.0;
            double vx = 0.0;
            double vy = 0.0;
            double boxScaleVel = 0.0;
            double confidence = 1.0;
        };

        int id = -1;
        cv::Rect2f box;
        cv::Point2f velocity = { 0.0f, 0.0f };
        cv::Point2f sizeVelocity = { 0.0f, 0.0f };
        int classId = -1;
        int hits = 0;
        int missed = 0;
        int age = 0;
        float confidence = 1.0f;
        bool observedThisFrame = false;
        double pivotX = 0.0;
        double pivotY = 0.0;
        double lastAssociationScore = 0.0;
        double lastAssociationDistancePx = 0.0;
        double lastAssociationIou = 0.0;
        double lastHeadingAlignment = 0.0;
        double lastNeuralScore = 0.5;
        double lastNeuralBonus = 0.0;
        bool lastNeuralEvaluated = false;
        InnerAimTrack innerAim;
        std::deque<TrackHistorySample> history;
        std::vector<std::pair<double, double>> temporalPrediction;
        int lastTemporalPredictionFrame = -1;
        int lastTemporalPredictionRequestFrame = -1;
        bool temporalPredictionValid = false;
        bool temporalPredictionPending = false;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    struct DetectionCandidate
    {
        cv::Rect2f box;
        int classId = -1;
        float confidence = 1.0f;
        double pivotX = 0.0;
        double pivotY = 0.0;
        double innerAimX = 0.0;
        double innerAimY = 0.0;
        int innerAimClassId = -1;
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    float scaleFactor() const;
    cv::Point2d computeInnerAimPoint(const cv::Rect2f& box, int classId) const;
    double computeAssociationCost(const cv::Rect& newBox, const InnerAimTrack& track, float newConf) const;
    aim::AimKalmanSettings buildInnerAimKalmanSettings(bool agileMotion) const;
    void initializeInnerAim(TrackState& t, const DetectionCandidate& d);
    void updateInnerAim(InnerAimTrack& track, const cv::Rect& det, float conf, double dt);
    void updateInnerAim(InnerAimTrack& track, const cv::Rect& det, float conf, double dt, double rawInnerX, double rawInnerY);
    void decayInnerAim(InnerAimTrack& track);
    aim::neural::TemporalPredictor::Input buildTemporalPredictorInput(
        const TrackState& t,
        int screenWidth,
        int screenHeight) const;
    void appendTrackHistory(TrackState& t);
    void updateTemporalPrediction(TrackState& t, int screenWidth, int screenHeight);
    bool shouldAcceptAsNewLock(const DetectionCandidate& det, const InnerAimTrack* current) const;
    int findTrackIndexById(int id) const;
    int chooseBestTrack(int screenWidth, int screenHeight) const;
    int allowedMissedFrames(const TrackState& t) const;
    void pruneDeadTracks();

    std::vector<TrackState> tracks_;
    int nextId_ = 1;
    int lockedTrackId_ = -1;
    int maxMissedFrames_ = 6;
    int updateFrameCounter_ = 0;
    std::mt19937 innerAimRng_{ std::random_device{}() };
};

#endif // BOXTARGET_H
