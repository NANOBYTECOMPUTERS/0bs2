import re
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "BoxTarget.h").exists()
)


class DetectionTrackingContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_detection_confidence_reaches_pid_observation(self):
        buffer_h = self.read("detector/detection_buffer.h")
        self.assertIn("std::vector<float> confidences", buffer_h)

        dml = self.read("detector/dml_detector.cpp")
        trt = self.read("detector/trt_detector.cpp")
        self.assertIn("confidences.push_back(d.confidence)", dml)
        self.assertIn("confidences.push_back(det.confidence)", trt)
        self.assertIn("detectionBuffer.publish(", dml)
        self.assertIn("detectionBuffer.publish(", trt)

        loop = self.read("runtime/mouse_thread_loop.cpp")
        self.assertIn("confidences = detectionBuffer.confidences", loop)
        self.assertIn("targetTracker.update(", loop)
        self.assertIn("mouseThread.moveMouseTarget(*activeTarget)", loop)
        self.assertIn("target.confidence", self.read("mouse/mouse.cpp"))

    def test_mouse_loop_has_direct_detection_fallback_without_overlay_dependency(self):
        loop = self.read("runtime/mouse_thread_loop.cpp")
        mouse_h = self.read("mouse/mouse.h")
        mouse_cpp = self.read("mouse/mouse.cpp")

        for token in (
            "chooseDirectDetectionTarget(",
            "boxes",
            "classes",
            "confidences",
            "trackerFrameWidth",
            "trackerFrameHeight",
            "BoxTarget target(",
            "target.smoothX = bestPivotX",
            "target.smoothY = bestPivotY",
            "mouseThread.moveMouseTarget(*activeTarget)",
        ):
            self.assertIn(token, loop)

        self.assertIn("void updateDetectionGeometry(int width, int height)", mouse_h)
        self.assertIn("void moveMouseTarget(const BoxTarget& target)", mouse_h)
        self.assertIn("MouseThread::updateDetectionGeometry", mouse_cpp)
        self.assertIn("center_x = screen_width * 0.5", mouse_cpp)
        self.assertIn("center_y = screen_height * 0.5", mouse_cpp)
        self.assertNotIn('#include "Game_overlay.h"', loop)
        self.assertNotIn("modelToScreenPoint", loop)

    def test_tracker_uses_motcpp_style_association_signals(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        self.assertIn("double confidence", target_h)
        self.assertIn("float confidence", target_h)
        self.assertIn("predictedBox", target_cpp)
        self.assertIn("confidenceBonus", target_cpp)
        self.assertIn("trackAssigned", target_cpp)

    def test_tracker_keeps_locked_identity_with_soft_motion_features(self):
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        self.assertIn("lastAssociationScore", target_h)
        self.assertIn("lastHeadingAlignment", target_h)
        self.assertIn("AssociationBreakdown", target_cpp)
        self.assertIn("headingPenalty", target_cpp)
        self.assertIn("lockedBias", target_cpp)

    def test_tracker_v2_uses_global_assignment_and_lifecycle_gates(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        target_h = self.read("mouse/BoxTarget.h")
        target_cpp = self.read("mouse/BoxTarget.cpp")

        for token in (
            "bool tracker_v2_enabled",
            "float tracker_v2_high_confidence",
            "float tracker_v2_new_track_confidence",
            "int tracker_v2_detector_max_candidates",
            "float tracker_v2_box_smoothing_alpha",
            "float tracker_v2_box_prediction_alpha",
        ):
            self.assertIn(token, config_h)

        for token in (
            'tracker_v2_enabled = get_bool("tracker_v2_enabled", true)',
            'MERGE_FIELD("tracker_v2_enabled", tracker_v2_enabled)',
            '"tracker_v2_high_confidence = " << tracker_v2_high_confidence',
            '"tracker_v2_detector_max_candidates = " << tracker_v2_detector_max_candidates',
        ):
            self.assertIn(token, config_cpp)

        for token in (
            "enum class TrackLifecycle",
            "struct BoxKalmanState",
            "BoxAxisKalman",
            "boxKalman",
            "stableBox",
            "stableBoxInitialized",
        ):
            self.assertIn(token, target_h)

        for token in (
            "solveHungarianSquare",
            "assignHungarianPass",
            "highConfidenceDetections",
            "lowConfidenceDetections",
            "kUnassignedAssociationCost",
            "predictBoxKalman",
            "correctBoxKalman",
            "TrackLifecycle::Tentative",
            "TrackLifecycle::Confirmed",
            "TrackLifecycle::Lost",
            "updateStableBox",
            "outputBoxForTrack",
            "suppressedByExistingTrackBox",
            "duplicateIouThreshold",
        ):
            self.assertIn(token, target_cpp)

    def test_inner_aim_propagates_missed_frames_without_kalman(self):
        target_cpp = self.read("mouse/BoxTarget.cpp")

        assignment_branch = target_cpp[target_cpp.index("if (di >= 0)"):]
        match = re.search(
            r"else\s*\{(?P<body>.*?)const float decay\s*=\s*\(t\.id == lockedTrackId_\)",
            assignment_branch,
            flags=re.S,
        )
        self.assertIsNotNone(match)
        missed_branch = match.group("body")
        self.assertIn("if (config.kalman_enabled && t.innerAim.kalman.initialized())", missed_branch)
        self.assertIn("else", missed_branch)
        self.assertIn("applyInnerAimEgoMotion(t.innerAim, compensatedEgoMotion)", missed_branch)
        self.assertIn("t.innerAim.smoothX += t.velocity.x * dt", missed_branch)
        self.assertIn("t.innerAim.smoothY += t.velocity.y * dt", missed_branch)


if __name__ == "__main__":
    unittest.main()
