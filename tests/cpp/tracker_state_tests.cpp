#include "BoxTarget.h"
#include "config.h"
#include "test_harness.h"

#include <algorithm>
#include <string>
#include <vector>

Config config;

namespace
{
void configureTrackerDefaults()
{
    config.class_player = 0;
    config.class_head = 1;
    config.disable_headshot = false;
    config.body_y_offset = Config::kBodyYOffsetDefault;
    config.head_y_offset = Config::kHeadYOffsetDefault;
    config.detection_resolution = 640;
    config.runtime_latency_sweep_enabled = false;
    config.estimator_mode = "kalman";
    config.kalman_enabled = false;
    config.kalman_max_velocity = 20000.0f;
    config.ego_motion_compensation_enabled = true;
    config.ego_motion_compensation_max_shift_px = 32.0f;
    config.tracker_v2_enabled = true;
    config.tracker_v2_high_confidence = 0.45f;
    config.tracker_v2_new_track_confidence = 0.55f;
    config.tracker_v2_detector_max_candidates = 160;
    config.tracker_v2_box_smoothing_alpha = 0.50f;
    config.tracker_v2_box_prediction_alpha = 0.20f;
}

LockedTargetInfo requireLockedTarget(const MultiTargetTracker& tracker)
{
    LockedTargetInfo info;
    REQUIRE(tracker.getLockedTarget(info));
    REQUIRE(info.trackId > 0);
    return info;
}

TrackDebugInfo requireSingleDebugTrack(const MultiTargetTracker& tracker)
{
    const std::vector<TrackDebugInfo> debugTracks = tracker.getDebugTracks();
    REQUIRE(debugTracks.size() == 1);
    return debugTracks[0];
}

void testTrackerLocksAndMaintainsIdThroughMatchedMovement()
{
    configureTrackerDefaults();
    MultiTargetTracker tracker;

    tracker.update(
        { cv::Rect(300, 200, 80, 180) },
        { config.class_player },
        { 0.92f },
        800,
        600,
        false,
        false);

    const LockedTargetInfo first = requireLockedTarget(tracker);
    REQUIRE(first.observedThisFrame);
    REQUIRE(first.missedFrames == 0);
    REQUIRE(first.target.classId == config.class_player);
    REQUIRE_NEAR(first.target.smoothX, 340.0, 1e-3);
    REQUIRE_NEAR(first.target.smoothY, 227.0, 1e-3);

    tracker.update(
        { cv::Rect(306, 202, 80, 180) },
        { config.class_player },
        { 0.88f },
        800,
        600,
        false,
        true);

    const LockedTargetInfo second = requireLockedTarget(tracker);
    REQUIRE(second.trackId == first.trackId);
    REQUIRE(second.observedThisFrame);
    REQUIRE(second.missedFrames == 0);
    REQUIRE_NEAR(second.target.smoothX, 346.0, 1e-3);
    REQUIRE_NEAR(second.target.smoothY, 229.0, 1e-3);

    const TrackDebugInfo debug = requireSingleDebugTrack(tracker);
    REQUIRE(debug.trackId == first.trackId);
    REQUIRE(debug.hits >= 2);
    REQUIRE(debug.lifecycle == 1);
    REQUIRE(debug.isLocked);
}

void testTrackerKeepsLockedTargetDuringShortOcclusion()
{
    configureTrackerDefaults();
    MultiTargetTracker tracker;

    tracker.update(
        { cv::Rect(300, 200, 80, 180) },
        { config.class_player },
        { 0.92f },
        800,
        600,
        false,
        false);
    tracker.update(
        { cv::Rect(306, 202, 80, 180) },
        { config.class_player },
        { 0.90f },
        800,
        600,
        false,
        true);

    const LockedTargetInfo observed = requireLockedTarget(tracker);

    tracker.update({}, {}, {}, 800, 600, false, true);

    const LockedTargetInfo occluded = requireLockedTarget(tracker);
    REQUIRE(occluded.trackId == observed.trackId);
    REQUIRE(!occluded.observedThisFrame);
    REQUIRE(occluded.missedFrames == 1);
    REQUIRE(occluded.trackHistory.size() >= 3);

    const TrackDebugInfo debug = requireSingleDebugTrack(tracker);
    REQUIRE(debug.trackId == observed.trackId);
    REQUIRE(debug.lifecycle == 2);
    REQUIRE(debug.missedFrames == 1);

    tracker.reset();
    LockedTargetInfo afterReset;
    REQUIRE(!tracker.getLockedTarget(afterReset));
    REQUIRE(tracker.getLockedTrackId() == -1);
    REQUIRE(tracker.getDebugTracks().empty());
}

void testTrackerGatesLowConfidenceAndPromotesTentativeTrack()
{
    configureTrackerDefaults();
    MultiTargetTracker tracker;

    tracker.update(
        { cv::Rect(310, 220, 70, 160) },
        { config.class_player },
        { 0.40f },
        800,
        600,
        false,
        false);

    REQUIRE(tracker.getDebugTracks().empty());
    LockedTargetInfo info;
    REQUIRE(!tracker.getLockedTarget(info));

    tracker.update(
        { cv::Rect(310, 220, 70, 160) },
        { config.class_player },
        { 0.70f },
        800,
        600,
        false,
        false);

    const LockedTargetInfo tentative = requireLockedTarget(tracker);
    TrackDebugInfo debug = requireSingleDebugTrack(tracker);
    REQUIRE(debug.trackId == tentative.trackId);
    REQUIRE(debug.lifecycle == 0);
    REQUIRE(debug.hits == 1);

    tracker.update(
        { cv::Rect(314, 222, 70, 160) },
        { config.class_player },
        { 0.69f },
        800,
        600,
        false,
        true);

    const LockedTargetInfo confirmed = requireLockedTarget(tracker);
    REQUIRE(confirmed.trackId == tentative.trackId);
    debug = requireSingleDebugTrack(tracker);
    REQUIRE(debug.lifecycle == 1);
    REQUIRE(debug.hits >= 2);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "tracker locks and maintains ID through matched movement", testTrackerLocksAndMaintainsIdThroughMatchedMovement },
            { "tracker keeps locked target during short occlusion", testTrackerKeepsLockedTargetDuringShortOcclusion },
            { "tracker gates low confidence and promotes tentative track", testTrackerGatesLowConfidenceAndPromotesTentativeTrack },
        },
        "tracker state");
}
