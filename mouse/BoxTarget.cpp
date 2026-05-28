#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <memory>
#include <string>
#include <utility>
#include <opencv2/opencv.hpp>

#include "0BS_box_2.h"
#include "BoxTarget.h"
#include "config.h"
#include "neural/NeuralTracker.h"
#include "neural/TemporalPredictor.h"

BoxTarget::BoxTarget()
    : x(0), y(0), w(0), h(0), classId(0), pivotX(0.0), pivotY(0.0), smoothX(0.0), smoothY(0.0), confidence(1.0), trackId(-1)
{
}

BoxTarget::BoxTarget(
    int x_,
    int y_,
    int w_,
    int h_,
    int cls,
    double px,
    double py,
    double conf,
    int tid)
    : x(x_), y(y_), w(w_), h(h_), classId(cls), pivotX(px), pivotY(py), smoothX(px), smoothY(py), confidence(conf), trackId(tid)
{
}

BoxTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot,
    const std::vector<float>* confidences)
{
    if (boxes.empty() || classes.empty())
    {
        return nullptr;
    }

    cv::Point center(screenWidth / 2, screenHeight / 2);

    double minDistance = std::numeric_limits<double>::max();
    int nearestIdx = -1;
    int targetY = 0;

    if (!disableHeadshot)
    {
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (classes[i] == config.class_head)
            {
                int headOffsetY = static_cast<int>(boxes[i].height * config.head_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + headOffsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (disableHeadshot || nearestIdx == -1)
    {
        minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (disableHeadshot && classes[i] == config.class_head)
                continue;

            if (classes[i] == config.class_player)
            {
                int offsetY = static_cast<int>(boxes[i].height * config.body_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + offsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (nearestIdx == -1)
    {
        return nullptr;
    }

    int finalY = 0;
    if (classes[nearestIdx] == config.class_head)
    {
        int headOffsetY = static_cast<int>(boxes[nearestIdx].height * config.head_y_offset);
        finalY = boxes[nearestIdx].y + headOffsetY - boxes[nearestIdx].height / 2;
    }
    else
    {
        finalY = targetY - boxes[nearestIdx].height / 2;
    }

    int finalX = boxes[nearestIdx].x;
    int finalW = boxes[nearestIdx].width;
    int finalH = boxes[nearestIdx].height;
    int finalClass = classes[nearestIdx];

    double pivotX = finalX + (finalW / 2.0);
    double pivotY = finalY + (finalH / 2.0);

    double finalConfidence = 1.0;
    if (confidences && nearestIdx >= 0 && static_cast<size_t>(nearestIdx) < confidences->size())
        finalConfidence = std::clamp(static_cast<double>((*confidences)[nearestIdx]), 0.0, 1.0);

    return new BoxTarget(finalX, finalY, finalW, finalH, finalClass, pivotX, pivotY, finalConfidence);
}

float MultiTargetTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float ua = a.width * a.height + b.width * b.height - inter;
    if (ua <= 1e-6f) return 0.0f;
    return inter / ua;
}

float MultiTargetTracker::scaleFactor() const
{
    return std::clamp(config.detection_resolution / 320.0f, 0.5f, 2.5f);
}

cv::Point2d MultiTargetTracker::computeInnerAimPoint(const cv::Rect2f& box, int classId) const
{
    const float smallBoxThreshold = 18.0f * scaleFactor();
    const float mediumBoxThreshold = 45.0f * scaleFactor();

    const double x = box.x + box.width * 0.5;
    double yBias = 0.5;
    if (classId == config.class_head)
    {
        yBias = std::clamp(static_cast<double>(config.head_y_offset), 0.05, 0.55);
    }
    else if (classId == config.class_player)
    {
        yBias = std::clamp(static_cast<double>(config.body_y_offset), 0.20, 0.90);
    }
    else if (box.width < smallBoxThreshold)
    {
        yBias = 0.5;
    }
    else if (box.width < mediumBoxThreshold)
    {
        yBias = std::clamp(static_cast<double>(config.head_y_offset), 0.10, 0.48);
    }
    else
    {
        (void)classId;
        yBias = 0.48;
    }

    return { x, box.y + box.height * yBias };
}

double MultiTargetTracker::computeAssociationCost(const cv::Rect& newBox, const InnerAimTrack& track, float newConf) const
{
    const cv::Rect2f newBoxF(
        static_cast<float>(newBox.x),
        static_cast<float>(newBox.y),
        static_cast<float>(newBox.width),
        static_cast<float>(newBox.height));
    const cv::Rect2f currentBoxF(
        static_cast<float>(track.outerBox.x),
        static_cast<float>(track.outerBox.y),
        static_cast<float>(track.outerBox.width),
        static_cast<float>(track.outerBox.height));

    const float iouScore = std::clamp(iou(newBoxF, currentBoxF), 0.0f, 1.0f);
    const cv::Point2d innerPoint = computeInnerAimPoint(newBoxF, track.classId);
    const double innerX = innerPoint.x;
    const double innerY = innerPoint.y;
    const double dist = std::hypot(innerX - track.smoothX, innerY - track.smoothY);
    const double normDist = std::min(1.0, dist / std::max(12.0, static_cast<double>(newBox.width)));
    const double consistencyBonus = std::clamp(static_cast<double>(track.consistencyScore * 0.8f), 0.0, 1.0);

    double cost =
        0.22 * (1.0f - iouScore) +
        0.48 * normDist +
        0.30 * (1.0 - consistencyBonus);

    const double proximity = 16.0f * scaleFactor();
    if (dist < proximity)
    {
        if (!(newConf > track.confidence + 0.38f && newConf > 0.72f))
            cost += 0.28 * (1.0 - dist / std::max(1.0, proximity));
    }

    return cost;
}

aim::AimKalmanSettings MultiTargetTracker::buildInnerAimKalmanSettings(bool agileMotion) const
{
    aim::AimKalmanSettings settings;
    settings.runtimeLatencySweepEnabled = config.runtime_latency_sweep_enabled;
    settings.enabled = config.kalman_enabled;
    settings.process_noise_position = agileMotion ? 2.2 : 0.6;
    settings.process_noise_velocity = agileMotion ? 3200.0 : 1400.0;
    settings.measurement_noise = std::max(0.85, static_cast<double>(1.15f * scaleFactor()));
    settings.velocity_damping = agileMotion ? 0.18 : 0.32;
    settings.max_velocity = std::clamp(static_cast<double>(config.kalman_max_velocity), 100.0, 60000.0);
    settings.warmup_frames = 2;
    settings.velocitySeedEnabled = true;
    settings.acquisitionFrames = 4;
    return settings;
}

void MultiTargetTracker::initializeInnerAim(TrackState& t, const DetectionCandidate& d)
{
    t.innerAim = InnerAimTrack();
    t.innerAim.trackId = t.id;
    t.innerAim.classId = d.innerAimClassId;
    const cv::Rect det(
        static_cast<int>(std::lround(d.box.x)),
        static_cast<int>(std::lround(d.box.y)),
        static_cast<int>(std::lround(d.box.width)),
        static_cast<int>(std::lround(d.box.height)));
    t.innerAim.hits = t.hits;
    updateInnerAim(t.innerAim, det, d.confidence, 1.0 / 120.0, d.innerAimX, d.innerAimY);
}

void MultiTargetTracker::updateInnerAim(InnerAimTrack& track, const cv::Rect& det, float conf, double dt)
{
    const cv::Point2d rawInner = computeInnerAimPoint(
        cv::Rect2f(
            static_cast<float>(det.x),
            static_cast<float>(det.y),
            static_cast<float>(det.width),
            static_cast<float>(det.height)),
        track.classId);
    updateInnerAim(track, det, conf, dt, rawInner.x, rawInner.y);
}

void MultiTargetTracker::updateInnerAim(InnerAimTrack& track, const cv::Rect& det, float conf, double dt, double rawInnerX, double rawInnerY)
{
    track.observedThisFrame = true;
    track.outerBox = det;
    track.confidence = std::clamp(conf, 0.0f, 1.0f);

    const double closeDistance = std::hypot(rawInnerX - track.smoothX, rawInnerY - track.smoothY);
    if (config.kalman_enabled && track.kalman.initialized() && closeDistance < 2.5 * scaleFactor() && track.consistencyScore > 0.70f)
    {
        const double jitterScale = 0.25 * (1.0 - std::clamp(static_cast<double>(track.consistencyScore), 0.0, 1.0));
        // Skip the RNG draw entirely when jitterScale collapses to zero (perfect consistency)
        // so the RNG sequence does not depend on how often this branch is reached.
        if (jitterScale > 0.0)
        {
            std::uniform_real_distribution<double> jitter(-0.4f * scaleFactor(), 0.4f * scaleFactor());
            rawInnerX += jitter(innerAimRng_) * jitterScale;
            rawInnerY += jitter(innerAimRng_) * jitterScale;
        }
    }

    if (!config.kalman_enabled)
    {
        track.kalman.reset();
        track.smoothX = rawInnerX;
        track.smoothY = rawInnerY;
    }
    else
    {
        const auto velocity = track.kalman.velocity();
        const bool highSpeed =
            std::abs(velocity.first) > 45.0 * scaleFactor() ||
            std::abs(velocity.second) > 45.0 * scaleFactor();
        track.kalman.setSettings(buildInnerAimKalmanSettings(highSpeed));

        const double lookahead = highSpeed ? 0.018 : 0.011;
        const aim::AimKalmanTelemetry telemetry = track.kalman.update(rawInnerX, rawInnerY, dt, lookahead);
        // Fast tracks keep the intended Kalman velocity * 0.011 forward bias, clamped to avoid orbiting.
        const double velocityLeadSeconds = highSpeed ? 0.011 : 0.0;
        double leadX = telemetry.velocity_x * velocityLeadSeconds;
        double leadY = telemetry.velocity_y * velocityLeadSeconds;
        const double maxLead = 2.5 * scaleFactor();
        const double leadMag = std::hypot(leadX, leadY);
        if (leadMag > maxLead && leadMag > 1e-9)
        {
            const double leadScale = maxLead / leadMag;
            leadX *= leadScale;
            leadY *= leadScale;
        }
        track.smoothX = telemetry.estimate_x + leadX;
        track.smoothY = telemetry.estimate_y + leadY;
    }

    track.smoothX = std::clamp(track.smoothX, det.x - 2.0 * scaleFactor(), det.x + det.width + 2.0 * scaleFactor());
    track.smoothY = std::clamp(track.smoothY, det.y - 2.0 * scaleFactor(), det.y + det.height + 2.0 * scaleFactor());

    track.radius = (4.0f + (1.0f - track.confidence) * 14.0f) * scaleFactor();
    if (config.kalman_enabled && track.kalman.initialized())
        track.radius = std::max(3.5f * scaleFactor(), track.radius * 0.65f);

    track.consistencyScore = std::min(1.0f, track.consistencyScore + 0.22f);
    track.missedFrames = 0;
}

void MultiTargetTracker::decayInnerAim(InnerAimTrack& track)
{
    if (!track.observedThisFrame)
    {
        track.consistencyScore = std::max(0.0f, track.consistencyScore - 0.085f);
        track.missedFrames++;
        track.radius = std::min(48.0f * scaleFactor(), track.radius + 2.0f * scaleFactor());
    }
    else
    {
        track.missedFrames = 0;
    }
}

aim::neural::TemporalPredictor::Input MultiTargetTracker::buildTemporalPredictorInput(
    const TrackState& t,
    int screenWidth,
    int screenHeight) const
{
    aim::neural::TemporalPredictor::Input input;
    input.history_length = std::clamp(config.temporal_prediction_history_length, 2, 64);

    if (t.history.empty())
        return input;

    const size_t needed = static_cast<size_t>(input.history_length);
    input.history.reserve(needed * aim::neural::TemporalPredictorFeatureCount);
    const size_t available = t.history.size();
    const size_t start = available > needed ? available - needed : 0;
    const auto& pad = t.history.front();
    const int padCount = static_cast<int>(needed - std::min(needed, available));

    auto appendSample = [&](const TrackState::TrackHistorySample& s)
        {
            input.history.push_back(static_cast<float>(s.x));
            input.history.push_back(static_cast<float>(s.y));
            input.history.push_back(static_cast<float>(s.w));
            input.history.push_back(static_cast<float>(s.h));
            input.history.push_back(static_cast<float>(s.vx));
            input.history.push_back(static_cast<float>(s.vy));
            input.history.push_back(static_cast<float>(s.boxScaleVel));
            input.history.push_back(static_cast<float>(std::clamp(s.confidence, 0.0, 1.0)));
        };

    for (int i = 0; i < padCount; ++i)
        appendSample(pad);

    for (size_t i = start; i < available; ++i)
        appendSample(t.history[i]);

    (void)screenWidth;
    (void)screenHeight;
    return input;
}

void MultiTargetTracker::appendTrackHistory(TrackState& t)
{
    TrackState::TrackHistorySample sample;
    sample.x = t.innerAim.smoothX;
    sample.y = t.innerAim.smoothY;
    sample.w = t.box.width;
    sample.h = t.box.height;
    sample.vx = t.velocity.x;
    sample.vy = t.velocity.y;
    sample.boxScaleVel = (static_cast<double>(t.sizeVelocity.x) + static_cast<double>(t.sizeVelocity.y)) * 0.5;
    sample.confidence = std::clamp(static_cast<double>(t.confidence), 0.0, 1.0);

    t.history.push_back(sample);
    const size_t maxHistory = static_cast<size_t>(std::clamp(config.temporal_prediction_history_length * 2, 8, 128));
    while (t.history.size() > maxHistory)
        t.history.pop_front();
}

void MultiTargetTracker::updateTemporalPrediction(TrackState& t, int screenWidth, int screenHeight)
{
    auto& worker = aim::neural::TemporalPredictionWorker::instance();

    if (!config.temporal_prediction_enabled)
    {
        t.temporalPrediction.clear();
        t.temporalPredictionValid = false;
        t.temporalPredictionPending = false;
        worker.clear();
        return;
    }

    const int historyLength = std::clamp(config.temporal_prediction_history_length, 2, 64);
    if (static_cast<int>(t.history.size()) < historyLength)
    {
        t.temporalPrediction.clear();
        t.temporalPredictionValid = false;
        t.temporalPredictionPending = false;
        return;
    }

    const int interval = std::clamp(config.temporal_prediction_interval_frames, 1, 16);
    aim::neural::TemporalPredictionWorker::Result workerResult;
    if (worker.tryGet(t.id, workerResult))
    {
        t.temporalPredictionPending = false;
        if (workerResult.valid &&
            workerResult.frame_id >= t.lastTemporalPredictionFrame &&
            !workerResult.output.future_x.empty() &&
            workerResult.output.future_x.size() == workerResult.output.future_y.size())
        {
            t.temporalPrediction.clear();
            t.temporalPrediction.reserve(workerResult.output.future_x.size());
            for (size_t i = 0; i < workerResult.output.future_x.size(); ++i)
            {
                const double x = std::clamp(static_cast<double>(workerResult.output.future_x[i]), 0.0, static_cast<double>(screenWidth));
                const double y = std::clamp(static_cast<double>(workerResult.output.future_y[i]), 0.0, static_cast<double>(screenHeight));
                t.temporalPrediction.emplace_back(x, y);
            }
            t.temporalPredictionValid = !t.temporalPrediction.empty();
            t.lastTemporalPredictionFrame = workerResult.frame_id;
        }
    }

    if (t.temporalPredictionValid &&
        updateFrameCounter_ - t.lastTemporalPredictionFrame < interval)
    {
        return;
    }

    if (t.temporalPredictionPending &&
        updateFrameCounter_ - t.lastTemporalPredictionRequestFrame < std::max(1, interval * 2))
    {
        return;
    }

    auto input = buildTemporalPredictorInput(t, screenWidth, screenHeight);
    aim::neural::TemporalPredictionWorker::Request request;
    request.track_id = t.id;
    request.frame_id = updateFrameCounter_;
    request.model_path = config.temporal_prediction_model_path;
    request.history_length = historyLength;
    request.prediction_horizon = std::clamp(config.temporal_prediction_horizon, 1, 64);
    request.input = std::move(input);
    worker.submit(request);
    t.temporalPredictionPending = true;
    t.lastTemporalPredictionRequestFrame = updateFrameCounter_;
}

bool MultiTargetTracker::shouldAcceptAsNewLock(const DetectionCandidate& det, const InnerAimTrack* current) const
{
    if (!current)
        return true;

    const double dist = std::hypot(det.pivotX - current->smoothX, det.pivotY - current->smoothY);
    if (dist < 16.0f * scaleFactor())
    {
        return det.confidence > current->confidence + 0.38f &&
               det.confidence > 0.72f;
    }

    return true;
}

int MultiTargetTracker::findTrackIndexById(int id) const
{
    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        if (tracks_[i].id == id)
            return static_cast<int>(i);
    }
    return -1;
}

int MultiTargetTracker::allowedMissedFrames(const TrackState& t) const
{
    // Keep the locked target alive longer to survive short occlusion/fast motion bursts.
    const int lockedBonus = (t.id == lockedTrackId_) ? 8 : 0;
    const int historyBonus = std::clamp(t.hits / 3, 0, 4);
    const int confidenceBonus = (t.confidence > 0.65f) ? 2 : 0;
    return maxMissedFrames_ + lockedBonus + historyBonus + confidenceBonus;
}

void MultiTargetTracker::pruneDeadTracks()
{
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackState& t) {
            return t.missed > allowedMissedFrames(t);
            }),
        tracks_.end());
}

int MultiTargetTracker::chooseBestTrack(int screenWidth, int screenHeight) const
{
    if (tracks_.empty())
        return -1;

    const double cx = screenWidth * 0.5;
    const double cy = screenHeight * 0.5;

    int bestIdx = -1;
    double bestScore = std::numeric_limits<double>::max();

    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        const auto& t = tracks_[i];
        if (t.missed > allowedMissedFrames(t))
            continue;

        const double dx = t.innerAim.smoothX - cx;
        const double dy = t.innerAim.smoothY - cy;
        const double dist = std::hypot(dx, dy);
        const double hitBonus = std::min(5, t.hits) * 4.0;
        const double confidenceBonus = std::clamp(static_cast<double>(t.confidence), 0.0, 1.0) * 12.0;
        const double observedBonus = t.observedThisFrame ? 8.0 : 0.0;
        const double missPenalty = t.missed * 50.0;
        const double score = dist + missPenalty - hitBonus - confidenceBonus - observedBonus;

        if (score < bestScore)
        {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        }
    }

    return bestIdx;
}

void MultiTargetTracker::reset()
{
    tracks_.clear();
    nextId_ = 1;
    lockedTrackId_ = -1;
    updateFrameCounter_ = 0;
    aim::neural::TemporalPredictionWorker::instance().clear();
}

void MultiTargetTracker::update(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot,
    bool keepCurrentLock)
{
    static const std::vector<float> defaultConfidences;
    update(boxes, classes, defaultConfidences, screenWidth, screenHeight, disableHeadshot, keepCurrentLock);
}

void MultiTargetTracker::update(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    const std::vector<float>& confidences,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot,
    bool keepCurrentLock)
{
    const auto now = std::chrono::steady_clock::now();
    ++updateFrameCounter_;

    for (auto& t : tracks_)
    {
        t.observedThisFrame = false;
        t.innerAim.observedThisFrame = false;
    }

    if (boxes.size() != classes.size())
    {
        pruneDeadTracks();
        return;
    }

    std::vector<DetectionCandidate> dets;
    dets.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const int cls = classes[i];
        if (disableHeadshot)
        {
            if (cls != config.class_player)
                continue;
        }
        else
        {
            if (cls != config.class_player && cls != config.class_head)
            {
                continue;
            }
        }

        const cv::Rect& b = boxes[i];
        DetectionCandidate d;
        d.box = cv::Rect2f(static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.width), static_cast<float>(b.height));
        d.classId = cls;
        d.confidence = (i < confidences.size()) ? std::clamp(confidences[i], 0.0f, 1.0f) : 1.0f;
        const cv::Point2d innerPoint = computeInnerAimPoint(d.box, cls);
        d.pivotX = innerPoint.x;
        d.pivotY = innerPoint.y;
        d.innerAimX = innerPoint.x;
        d.innerAimY = innerPoint.y;
        d.innerAimClassId = cls;
        dets.push_back(d);
    }

    if (!disableHeadshot && !dets.empty())
    {
        // If head and player detections refer to the same entity, keep one track candidate
        // (player box for stable identity) but move its pivot to the head point.
        std::vector<size_t> playerIdx;
        playerIdx.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); ++i)
        {
            if (dets[i].classId == config.class_player)
                playerIdx.push_back(i);
        }

        if (!playerIdx.empty())
        {
            std::vector<char> dropHead(dets.size(), 0);
            std::vector<char> playerHasHeadPivot(dets.size(), 0);
            std::vector<double> playerHeadPivotX(dets.size(), 0.0);
            std::vector<double> playerHeadPivotY(dets.size(), 0.0);
            std::vector<double> playerHeadPivotDist(dets.size(), std::numeric_limits<double>::max());

            for (size_t hi = 0; hi < dets.size(); ++hi)
            {
                const auto& h = dets[hi];
                if (h.classId != config.class_head)
                    continue;

                const double headCx = h.box.x + h.box.width * 0.5;
                const double headCy = h.box.y + h.box.height * 0.5;

                size_t bestPlayer = static_cast<size_t>(-1);
                double bestDist = std::numeric_limits<double>::max();

                for (size_t pi : playerIdx)
                {
                    const auto& p = dets[pi].box;
                    const double px1 = p.x - p.width * 0.15;
                    const double px2 = p.x + p.width * 1.15;
                    const double py1 = p.y - p.height * 0.20;
                    const double py2 = p.y + p.height * 0.65;

                    if (!(headCx >= px1 && headCx <= px2 && headCy >= py1 && headCy <= py2))
                        continue;

                    const double pCx = p.x + p.width * 0.5;
                    const double pCy = p.y + p.height * 0.5;
                    const double d = std::hypot(headCx - pCx, headCy - pCy);
                    if (d < bestDist)
                    {
                        bestDist = d;
                        bestPlayer = pi;
                    }
                }

                if (bestPlayer != static_cast<size_t>(-1))
                {
                    dropHead[hi] = 1;
                    if (!playerHasHeadPivot[bestPlayer] || bestDist < playerHeadPivotDist[bestPlayer])
                    {
                        playerHasHeadPivot[bestPlayer] = 1;
                        playerHeadPivotDist[bestPlayer] = bestDist;
                        playerHeadPivotX[bestPlayer] = h.box.x + h.box.width * 0.5;
                        playerHeadPivotY[bestPlayer] = h.box.y + h.box.height * config.head_y_offset;
                        dets[bestPlayer].confidence = std::max(dets[bestPlayer].confidence, h.confidence);
                    }
                }
            }

            std::vector<DetectionCandidate> filtered;
            filtered.reserve(dets.size());

            for (size_t i = 0; i < dets.size(); ++i)
            {
                if (dropHead[i])
                    continue;

                DetectionCandidate d = dets[i];
                if (d.classId == config.class_player && playerHasHeadPivot[i])
                {
                    d.pivotX = playerHeadPivotX[i];
                    d.pivotY = playerHeadPivotY[i];
                    d.innerAimX = playerHeadPivotX[i];
                    d.innerAimY = playerHeadPivotY[i];
                    d.innerAimClassId = config.class_head;
                }
                filtered.push_back(d);
            }

            dets.swap(filtered);
        }
    }

    std::vector<int> detAssigned(dets.size(), -1);
    std::vector<int> trackAssigned(tracks_.size(), -1);

    struct AssociationBreakdown
    {
        double score = std::numeric_limits<double>::infinity();
        double distancePx = 0.0;
        double maxDistancePx = 0.0;
        double overlap = 0.0;
        double sizePenalty = 0.0;
        double headingAlignment = 0.0;
        double headingPenalty = 0.0;
        double lockedBias = 0.0;
        double confidenceBonus = 0.0;
        double innerAssociationCost = 0.0;
        double antiStealPenalty = 0.0;
        double neuralScore = 0.5;
        double neuralBonus = 0.0;
        bool neuralEvaluated = false;
        aim::neural::NeuralTrackerFeatures neuralFeatures;
        bool accepted = false;
    };

    auto getNeuralTracker = []() -> std::shared_ptr<aim::neural::INeuralTracker>
        {
            static std::shared_ptr<aim::neural::INeuralTracker> tracker;
            static std::string loadedPath;
            static std::string loadedRuntime;
            static bool attemptedLoad = false;

            if (!config.neural_tracker_enabled)
            {
                tracker.reset();
                loadedPath.clear();
                loadedRuntime.clear();
                attemptedLoad = false;
                return nullptr;
            }

            if (!attemptedLoad
                || loadedPath != config.neural_tracker_model_path
                || loadedRuntime != config.neural_tracker_runtime)
            {
                loadedPath = config.neural_tracker_model_path;
                loadedRuntime = config.neural_tracker_runtime;
                tracker = aim::neural::createNeuralTracker(loadedPath, loadedRuntime);
                attemptedLoad = true;
            }

            return tracker;
        };

    auto buildNeuralFeatures = [&](const TrackState& t,
                                   const DetectionCandidate& d,
                                   const cv::Rect2f& predictedBox,
                                   double dt,
                                   double dist,
                                   double maxDist,
                                   double overlap,
                                   double headingAlignment,
                                   bool classCompatible,
                                   bool relaxedForLocked) -> aim::neural::NeuralTrackerFeatures
        {
            const double predictedArea = std::max(1.0, static_cast<double>(predictedBox.width * predictedBox.height));
            const double detArea = std::max(1.0, static_cast<double>(d.box.width * d.box.height));
            const double predCx = predictedBox.x + predictedBox.width * 0.5;
            const double predCy = predictedBox.y + predictedBox.height * 0.5;
            const double speed = std::hypot(t.velocity.x, t.velocity.y);
            const double screenScale = std::max(1.0, static_cast<double>(std::max(screenWidth, screenHeight)));

            aim::neural::NeuralTrackerFeatures features;
            features.distanceNorm = static_cast<float>(std::clamp(dist / std::max(1.0, maxDist), 0.0, 3.0));
            features.iou = static_cast<float>(std::clamp(overlap, 0.0, 1.0));
            features.sizeLogRatio = static_cast<float>(std::clamp(std::log(detArea / predictedArea), -3.0, 3.0));
            features.detectionConfidence = std::clamp(d.confidence, 0.0f, 1.0f);
            features.trackConfidence = std::clamp(t.confidence, 0.0f, 1.0f);
            features.headingAlignment = static_cast<float>(std::clamp(headingAlignment, -1.0, 1.0));
            features.trackMissedNorm = static_cast<float>(std::clamp(t.missed / 12.0, 0.0, 1.0));
            features.trackHitsNorm = static_cast<float>(std::clamp(t.hits / 12.0, 0.0, 1.0));
            features.isLocked = (t.id == lockedTrackId_) ? 1.0f : 0.0f;
            features.classCompatible = classCompatible ? 1.0f : 0.0f;
            features.dt = static_cast<float>(std::clamp(dt, 0.0, 0.25));
            features.speedNorm = static_cast<float>(std::clamp(speed / (screenScale * 3.5), 0.0, 2.0));
            features.targetSizeNorm = static_cast<float>(std::clamp(std::sqrt(detArea) / screenScale, 0.0, 1.0));
            features.pivotOffsetXNorm = static_cast<float>(std::clamp((d.pivotX - predCx) / screenScale, -2.0, 2.0));
            features.pivotOffsetYNorm = static_cast<float>(std::clamp((d.pivotY - predCy) / screenScale, -2.0, 2.0));
            features.relaxedGate = relaxedForLocked ? 1.0f : 0.0f;
            return features;
        };

    auto computeAssociationBreakdown = [&](const TrackState& t, const DetectionCandidate& d, bool relaxedForLocked) -> AssociationBreakdown
        {
            AssociationBreakdown breakdown;

            const bool sameClass = (d.classId == t.classId);
            double classPenalty = 0.0;
            if (!sameClass)
            {
                const bool allowHeadBodySwap =
                    !disableHeadshot &&
                    ((t.classId == config.class_player && d.classId == config.class_head) ||
                     (t.classId == config.class_head && d.classId == config.class_player));
                if (!allowHeadBodySwap)
                    return breakdown;

                classPenalty = 0.18;
            }

            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                1e-4, 0.25
            );

            const float predCx = t.box.x + t.box.width * 0.5f + t.velocity.x * static_cast<float>(dt);
            const float predCy = t.box.y + t.box.height * 0.5f + t.velocity.y * static_cast<float>(dt);
            const float predW = std::max(1.0f, t.box.width + t.sizeVelocity.x * static_cast<float>(dt));
            const float predH = std::max(1.0f, t.box.height + t.sizeVelocity.y * static_cast<float>(dt));
            cv::Rect2f predictedBox(predCx - predW * 0.5f, predCy - predH * 0.5f, predW, predH);

            const double detCx = d.box.x + d.box.width * 0.5;
            const double detCy = d.box.y + d.box.height * 0.5;
            const double dist = std::hypot(detCx - predCx, detCy - predCy);

            const double diag = std::hypot(static_cast<double>(predictedBox.width), static_cast<double>(predictedBox.height));
            const double speed = std::hypot(t.velocity.x, t.velocity.y);
            const double baseGate = std::max(24.0, diag * 1.15 + 10.0);
            const double speedGate = speed * dt * (1.8 + t.missed * 0.35);
            const double missGate = t.missed * std::max(14.0, diag * 0.18);
            double maxDist = baseGate + speedGate + missGate;
            if (relaxedForLocked)
                maxDist *= 1.6;

            breakdown.distancePx = dist;
            breakdown.maxDistancePx = maxDist;

            if (dist > maxDist)
                return breakdown;

            const double overlap = iou(predictedBox, d.box);
            const cv::Rect detRect(
                static_cast<int>(std::lround(d.box.x)),
                static_cast<int>(std::lround(d.box.y)),
                static_cast<int>(std::lround(d.box.width)),
                static_cast<int>(std::lround(d.box.height)));
            const double innerAssociationCost = computeAssociationCost(detRect, t.innerAim, d.confidence);
            const double predictedArea = std::max(1.0, static_cast<double>(predictedBox.width * predictedBox.height));
            const double detArea = std::max(1.0, static_cast<double>(d.box.width * d.box.height));
            const double sizePenalty = std::min(0.35, std::abs(std::log(detArea / predictedArea)) * 0.18);
            const double confidenceBonus = std::clamp(static_cast<double>(d.confidence), 0.0, 1.0) * 0.08;
            const double missPenalty = t.missed * 0.025;
            const double hitBonus = std::min(6, t.hits) * 0.01;

            const double prevCx = t.box.x + t.box.width * 0.5;
            const double prevCy = t.box.y + t.box.height * 0.5;
            const double moveX = detCx - prevCx;
            const double moveY = detCy - prevCy;
            const double moveMag = std::hypot(moveX, moveY);
            double headingAlignment = 0.0;
            double headingPenalty = 0.0;
            if (speed > 80.0 && moveMag > std::max(3.0, diag * 0.08))
            {
                headingAlignment = std::clamp(
                    (moveX * t.velocity.x + moveY * t.velocity.y) / (moveMag * speed),
                    -1.0,
                    1.0
                );
                const double directionWeight = (t.missed > 0) ? 0.08 : 0.04;
                headingPenalty = std::max(0.0, 0.10 - headingAlignment) * directionWeight;
                if (relaxedForLocked)
                    headingPenalty *= 0.55;
            }

            double lockedBias = 0.0;
            if (t.id == lockedTrackId_)
            {
                lockedBias = relaxedForLocked ? 0.16 : 0.08;
                lockedBias += std::min(0.06, static_cast<double>(std::max(0, t.hits - 1)) * 0.008);
                if (t.missed > 0)
                    lockedBias *= 0.75;
            }

            double antiStealPenalty = 0.0;
            if (lockedTrackId_ != -1 && t.id != lockedTrackId_)
            {
                const int lockedIdx = findTrackIndexById(lockedTrackId_);
                if (lockedIdx >= 0)
                {
                    const auto& locked = tracks_[lockedIdx].innerAim;
                    const double lockedProximity = std::hypot(d.pivotX - locked.smoothX, d.pivotY - locked.smoothY);
                    const double antiStealRadius = 16.0f * scaleFactor();
                    if (lockedProximity < antiStealRadius)
                    {
                        if (!shouldAcceptAsNewLock(d, &locked))
                            antiStealPenalty = 3.0;
                        else
                            antiStealPenalty = 0.18 * (1.0 - lockedProximity / std::max(1.0, antiStealRadius));
                    }
                }
            }

            breakdown.overlap = overlap;
            breakdown.sizePenalty = sizePenalty;
            breakdown.headingAlignment = headingAlignment;
            breakdown.headingPenalty = headingPenalty;
            breakdown.lockedBias = lockedBias;
            breakdown.confidenceBonus = confidenceBonus;
            breakdown.innerAssociationCost = innerAssociationCost;
            breakdown.antiStealPenalty = antiStealPenalty;
            if (antiStealPenalty >= 3.0)
                return breakdown;

            breakdown.accepted = true;
            breakdown.score =
                innerAssociationCost +
                std::clamp(dist / maxDist, 0.0, 1.5) * 0.35 +
                sizePenalty +
                classPenalty +
                missPenalty +
                headingPenalty -
                hitBonus -
                confidenceBonus -
                lockedBias +
                antiStealPenalty;

            if (config.neural_tracker_enabled)
            {
                breakdown.neuralFeatures = buildNeuralFeatures(
                    t,
                    d,
                    predictedBox,
                    dt,
                    dist,
                    maxDist,
                    overlap,
                    headingAlignment,
                    true,
                    relaxedForLocked);

                if (auto tracker = getNeuralTracker())
                {
                    const aim::neural::NeuralTrackerResult neuralResult = tracker->score(breakdown.neuralFeatures);
                    if (neuralResult.valid)
                    {
                        breakdown.neuralScore = neuralResult.neuralScore;
                        breakdown.neuralEvaluated = true;
                        const double blend = std::clamp(static_cast<double>(config.neural_tracker_blend), 0.0, 1.0);
                        breakdown.neuralBonus = std::clamp((breakdown.neuralScore - 0.5) * 0.40 * blend, -0.20, 0.20);
                        breakdown.score -= breakdown.neuralBonus;
                    }
                }

                if (config.neural_tracker_log_enabled)
                {
                    aim::neural::logNeuralTrackerAssociation(
                        config.neural_tracker_log_path,
                        breakdown.neuralFeatures,
                        static_cast<float>(breakdown.neuralScore),
                        static_cast<float>(breakdown.score + breakdown.neuralBonus),
                        static_cast<float>(breakdown.score),
                        true,
                        false);
                }
            }
            return breakdown;
        };

    auto computeMatchScore = [&](const TrackState& t, const DetectionCandidate& d, bool relaxedForLocked) -> double
        {
            const AssociationBreakdown breakdown = computeAssociationBreakdown(t, d, relaxedForLocked);
            return breakdown.accepted ? breakdown.score : std::numeric_limits<double>::infinity();
        };

    auto tryAssignTrack = [&](int trackIndex, bool relaxedForLocked)
        {
            if (trackIndex < 0 || trackIndex >= static_cast<int>(tracks_.size()))
                return;
            if (trackAssigned[trackIndex] != -1)
                return;

            double bestScore = std::numeric_limits<double>::infinity();
            int bestDet = -1;
            const auto& track = tracks_[trackIndex];

            for (size_t di = 0; di < dets.size(); ++di)
            {
                if (detAssigned[di] != -1)
                    continue;

                const double score = computeMatchScore(track, dets[di], relaxedForLocked);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestDet = static_cast<int>(di);
                }
            }

            if (bestDet >= 0)
            {
                trackAssigned[trackIndex] = bestDet;
                detAssigned[bestDet] = trackIndex;
            }
        };

    // Always try to keep the locked track on the same identity first.
    if (lockedTrackId_ != -1)
    {
        const int lockedIdx = findTrackIndexById(lockedTrackId_);
        if (lockedIdx >= 0)
            tryAssignTrack(lockedIdx, true);
    }

    while (true)
    {
        double bestScore = std::numeric_limits<double>::max();
        int bestTi = -1;
        int bestDi = -1;

        for (size_t ti = 0; ti < tracks_.size(); ++ti)
        {
            if (trackAssigned[ti] != -1)
                continue;

            const auto& t = tracks_[ti];

            for (size_t di = 0; di < dets.size(); ++di)
            {
                if (detAssigned[di] != -1)
                    continue;
                const auto& d = dets[di];
                const double score = computeMatchScore(t, d, false);
                if (!std::isfinite(score))
                    continue;

                if (score < bestScore)
                {
                    bestScore = score;
                    bestTi = static_cast<int>(ti);
                    bestDi = static_cast<int>(di);
                }
            }
        }

        if (bestTi < 0 || bestDi < 0)
            break;

        trackAssigned[bestTi] = bestDi;
        detAssigned[bestDi] = bestTi;
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti)
    {
        auto& t = tracks_[ti];
        const int di = trackAssigned[ti];

        if (di >= 0)
        {
            const auto& d = dets[di];
            const AssociationBreakdown scoreBreakdown = computeAssociationBreakdown(
                t,
                d,
                t.id == lockedTrackId_
            );
            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                1e-4, 0.2
            );

            const float oldCx = t.box.x + t.box.width * 0.5f;
            const float oldCy = t.box.y + t.box.height * 0.5f;
            const float newCx = d.box.x + d.box.width * 0.5f;
            const float newCy = d.box.y + d.box.height * 0.5f;
            const cv::Point2f rawVel(
                static_cast<float>((newCx - oldCx) / dt),
                static_cast<float>((newCy - oldCy) / dt)
            );
            const cv::Point2f rawSizeVel(
                static_cast<float>((d.box.width - t.box.width) / dt),
                static_cast<float>((d.box.height - t.box.height) / dt)
            );

            cv::Point2f clampedRawVel = rawVel;
            const double rawSpeed = std::hypot(clampedRawVel.x, clampedRawVel.y);
            const double maxReasonableSpeed = std::max(screenWidth, screenHeight) * 3.5;
            if (rawSpeed > maxReasonableSpeed && rawSpeed > 1e-4)
            {
                const float scale = static_cast<float>(maxReasonableSpeed / rawSpeed);
                clampedRawVel *= scale;
            }

            const float blend = (t.id == lockedTrackId_) ? 0.45f : 0.35f;
            t.velocity = t.velocity * (1.0f - blend) + clampedRawVel * blend;
            t.sizeVelocity = t.sizeVelocity * (1.0f - blend) + rawSizeVel * blend;
            t.box = d.box;
            t.pivotX = d.pivotX;
            t.pivotY = d.pivotY;
            t.classId = d.classId;
            t.confidence = t.confidence * 0.35f + d.confidence * 0.65f;
            t.lastAssociationScore = scoreBreakdown.score;
            t.lastAssociationDistancePx = scoreBreakdown.distancePx;
            t.lastAssociationIou = scoreBreakdown.overlap;
            t.lastHeadingAlignment = scoreBreakdown.headingAlignment;
            t.lastNeuralScore = scoreBreakdown.neuralScore;
            t.lastNeuralBonus = scoreBreakdown.neuralBonus;
            t.lastNeuralEvaluated = scoreBreakdown.neuralEvaluated;
            if (config.neural_tracker_log_enabled && config.neural_tracker_enabled)
            {
                aim::neural::logNeuralTrackerAssociation(
                    config.neural_tracker_log_path,
                    scoreBreakdown.neuralFeatures,
                    static_cast<float>(scoreBreakdown.neuralScore),
                    static_cast<float>(scoreBreakdown.score + scoreBreakdown.neuralBonus),
                    static_cast<float>(scoreBreakdown.score),
                    scoreBreakdown.accepted,
                    true);
            }
            t.hits += 1;
            t.age += 1;
            t.missed = 0;
            t.observedThisFrame = true;
            t.innerAim.trackId = t.id;
            t.innerAim.classId = d.innerAimClassId;
            t.innerAim.hits = t.hits;
            updateInnerAim(
                t.innerAim,
                cv::Rect(
                    static_cast<int>(std::lround(d.box.x)),
                    static_cast<int>(std::lround(d.box.y)),
                    static_cast<int>(std::lround(d.box.width)),
                    static_cast<int>(std::lround(d.box.height))),
                d.confidence,
                dt,
                d.innerAimX,
                d.innerAimY);
            t.innerAim.hits = t.hits;
            t.lastUpdate = now;
        }
        else
        {
            const double dt = std::clamp(
                std::chrono::duration<double>(now - t.lastUpdate).count(),
                0.0, 0.2
            );
            t.box.x += t.velocity.x * static_cast<float>(dt);
            t.box.y += t.velocity.y * static_cast<float>(dt);
            t.box.width = std::max(1.0f, t.box.width + t.sizeVelocity.x * static_cast<float>(dt));
            t.box.height = std::max(1.0f, t.box.height + t.sizeVelocity.y * static_cast<float>(dt));
            t.pivotX += t.velocity.x * dt;
            t.pivotY += t.velocity.y * dt;
            if (config.kalman_enabled && t.innerAim.kalman.initialized())
            {
                const auto predicted = t.innerAim.kalman.predict(std::clamp(dt + 0.011, 0.0, 0.08));
                t.innerAim.smoothX = predicted.first;
                t.innerAim.smoothY = predicted.second;
            }
            else if (config.kalman_enabled)
            {
                t.innerAim.smoothX += t.velocity.x * dt;
                t.innerAim.smoothY += t.velocity.y * dt;
            }
            const float decay = (t.id == lockedTrackId_) ? 0.90f : 0.84f;
            t.velocity *= decay;
            t.sizeVelocity *= decay;
            t.confidence *= (t.id == lockedTrackId_) ? 0.88f : 0.80f;
            t.innerAim.confidence *= (t.id == lockedTrackId_) ? 0.90f : 0.82f;
            t.innerAim.outerBox = cv::Rect(
                static_cast<int>(std::lround(t.box.x)),
                static_cast<int>(std::lround(t.box.y)),
                static_cast<int>(std::lround(t.box.width)),
                static_cast<int>(std::lround(t.box.height)));
            t.innerAim.hits = t.hits;
            decayInnerAim(t.innerAim);
            t.lastAssociationScore = std::numeric_limits<double>::infinity();
            t.lastAssociationDistancePx = 0.0;
            t.lastAssociationIou = 0.0;
            t.lastNeuralScore = 0.5;
            t.lastNeuralBonus = 0.0;
            t.lastNeuralEvaluated = false;
            t.missed += 1;
            t.age += 1;
            t.observedThisFrame = false;
            t.lastUpdate = now;
        }

        appendTrackHistory(t);
        updateTemporalPrediction(t, screenWidth, screenHeight);
    }

    auto suppressedByLockedInnerAim = [&](const DetectionCandidate& d) -> bool
        {
            if (lockedTrackId_ == -1)
                return false;

            const int lockedIdx = findTrackIndexById(lockedTrackId_);
            if (lockedIdx < 0)
                return false;

            const InnerAimTrack& locked = tracks_[lockedIdx].innerAim;
            const double distance = std::hypot(d.pivotX - locked.smoothX, d.pivotY - locked.smoothY);
            if (distance >= 16.0f * scaleFactor())
                return false;

            return !shouldAcceptAsNewLock(d, &locked);
        };

    for (size_t di = 0; di < dets.size(); ++di)
    {
        if (detAssigned[di] != -1)
            continue;

        const auto& d = dets[di];
        if (suppressedByLockedInnerAim(d))
            continue;

        TrackState t;
        t.id = nextId_++;
        t.box = d.box;
        t.classId = d.classId;
        t.hits = 1;
        t.missed = 0;
        t.age = 1;
        t.confidence = d.confidence;
        t.observedThisFrame = true;
        t.pivotX = d.pivotX;
        t.pivotY = d.pivotY;
        t.lastAssociationScore = 0.0;
        t.lastAssociationDistancePx = 0.0;
        t.lastAssociationIou = 1.0;
        t.lastHeadingAlignment = 1.0;
        t.lastNeuralScore = 0.5;
        t.lastNeuralBonus = 0.0;
        t.lastNeuralEvaluated = false;
        t.lastUpdate = now;
        initializeInnerAim(t, d);
        appendTrackHistory(t);
        updateTemporalPrediction(t, screenWidth, screenHeight);
        tracks_.push_back(t);
    }

    pruneDeadTracks();

    if (findTrackIndexById(lockedTrackId_) < 0)
        lockedTrackId_ = -1;

    if (!keepCurrentLock)
    {
        const int bestIdx = chooseBestTrack(screenWidth, screenHeight);
        lockedTrackId_ = (bestIdx >= 0) ? tracks_[bestIdx].id : -1;
        return;
    }

    if (lockedTrackId_ == -1)
    {
        const int bestIdx = chooseBestTrack(screenWidth, screenHeight);
        lockedTrackId_ = (bestIdx >= 0) ? tracks_[bestIdx].id : -1;
    }
}

bool MultiTargetTracker::getLockedTarget(LockedTargetInfo& out) const
{
    const int idx = findTrackIndexById(lockedTrackId_);
    if (idx < 0)
        return false;

    const auto& t = tracks_[idx];
    if (t.missed > allowedMissedFrames(t))
        return false;

    out.trackId = t.id;
    out.observedThisFrame = t.observedThisFrame;
    out.missedFrames = t.missed;
    out.target = BoxTarget(
        static_cast<int>(std::lround(t.box.x)),
        static_cast<int>(std::lround(t.box.y)),
        static_cast<int>(std::lround(t.box.width)),
        static_cast<int>(std::lround(t.box.height)),
        t.classId,
        t.pivotX,
        t.pivotY,
        std::clamp(static_cast<double>(t.confidence), 0.0, 1.0),
        t.id
    );
    out.target.smoothX = t.innerAim.smoothX;
    out.target.smoothY = t.innerAim.smoothY;
    out.predictedFuture = t.temporalPredictionValid
        ? t.temporalPrediction
        : std::vector<std::pair<double, double>>{};
    out.predictedFutureAgeFrames = t.temporalPredictionValid
        ? std::max(0, updateFrameCounter_ - t.lastTemporalPredictionFrame)
        : 9999;
    out.targetVelocityX = t.velocity.x;
    out.targetVelocityY = t.velocity.y;
    out.targetBoxScaleVelocity = (static_cast<double>(t.sizeVelocity.x) + static_cast<double>(t.sizeVelocity.y)) * 0.5;
    return true;
}

std::vector<TrackDebugInfo> MultiTargetTracker::getDebugTracks() const
{
    std::vector<TrackDebugInfo> out;
    out.reserve(tracks_.size());

    for (const auto& t : tracks_)
    {
        if (t.missed > allowedMissedFrames(t))
            continue;

        TrackDebugInfo d;
        d.trackId = t.id;
        d.classId = t.classId;
        d.box = cv::Rect(
            static_cast<int>(std::lround(t.box.x)),
            static_cast<int>(std::lround(t.box.y)),
            static_cast<int>(std::lround(t.box.width)),
            static_cast<int>(std::lround(t.box.height))
        );
        d.pivotX = t.pivotX;
        d.pivotY = t.pivotY;
        d.innerAimX = t.innerAim.smoothX;
        d.innerAimY = t.innerAim.smoothY;
        d.innerAimRadius = t.innerAim.radius;
        d.consistencyScore = t.innerAim.consistencyScore;
        d.confidence = t.confidence;
        d.hits = t.hits;
        d.observedThisFrame = t.observedThisFrame;
        d.missedFrames = t.missed;
        d.isLocked = (t.id == lockedTrackId_);
        d.lastAssociationScore = t.lastAssociationScore;
        d.lastAssociationDistancePx = t.lastAssociationDistancePx;
        d.lastAssociationIou = t.lastAssociationIou;
        d.lastHeadingAlignment = t.lastHeadingAlignment;
        d.lastNeuralScore = t.lastNeuralScore;
        d.lastNeuralBonus = t.lastNeuralBonus;
        d.lastNeuralEvaluated = t.lastNeuralEvaluated;
        d.temporalFuture = t.temporalPrediction;
        d.temporalPredictionValid = t.temporalPredictionValid;
        out.push_back(d);
    }

    return out;
}
