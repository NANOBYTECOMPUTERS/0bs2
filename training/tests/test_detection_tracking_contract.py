import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


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
        self.assertIn("activeTarget->confidence", loop)

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


if __name__ == "__main__":
    unittest.main()
