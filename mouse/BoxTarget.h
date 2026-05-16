#ifndef BOXTARGET_H
#define BOXTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

class BoxTarget
{
public:
    BoxTarget();
    int x, y, w, h;
    int classId;

    double pivotX;
    double pivotY;
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

struct LockedTargetInfo
{
    int trackId = -1;
    bool observedThisFrame = false;
    int missedFrames = 0;
    BoxTarget target;
};

struct TrackDebugInfo
{
    int trackId = -1;
    int classId = -1;
    cv::Rect box;
    double pivotX = 0.0;
    double pivotY = 0.0;
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
        std::chrono::steady_clock::time_point lastUpdate;
    };

    struct DetectionCandidate
    {
        cv::Rect2f box;
        int classId = -1;
        float confidence = 1.0f;
        double pivotX = 0.0;
        double pivotY = 0.0;
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    int findTrackIndexById(int id) const;
    int chooseBestTrack(int screenWidth, int screenHeight) const;
    int allowedMissedFrames(const TrackState& t) const;
    void pruneDeadTracks();

    std::vector<TrackState> tracks_;
    int nextId_ = 1;
    int lockedTrackId_ = -1;
    int maxMissedFrames_ = 6;
};

#endif // BOXTARGET_H
