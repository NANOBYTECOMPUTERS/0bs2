import re
import unittest
from pathlib import Path

from training.pid_governor.dataset import FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[4]


class PidGovernorCppContractTests(unittest.TestCase):
    def test_cpp_feature_vector_matches_training_feature_order(self):
        cpp_path = REPO_ROOT / "mouse/PidGovernor.cpp"
        source = cpp_path.read_text(encoding="utf-8")
        body_match = re.search(r"std::array<float, PidGovernorFeatureCount> pidGovernorFeatures.*?return\s*\{(?P<body>.*?)\};", source, re.S)
        self.assertIsNotNone(body_match)

        names = re.findall(r"feature\(input\.([A-Za-z0-9_]+)\)", body_match.group("body"))
        expected = [
            "errorX",
            "errorY",
            "errorDistance",
            "errorDirectionRight",
            "errorDirectionDownRight",
            "errorDirectionDown",
            "errorDirectionDownLeft",
            "errorDirectionLeft",
            "errorDirectionUpLeft",
            "errorDirectionUp",
            "errorDirectionUpRight",
            "targetWidth",
            "targetHeight",
            "targetSize",
            "targetVx",
            "targetVy",
            "targetAx",
            "targetAy",
            "targetMotionStill",
            "targetMotionMoving",
            "cursorVx",
            "cursorVy",
            "previousOutputX",
            "previousOutputY",
            "pidPX",
            "pidPY",
            "pidIX",
            "pidIY",
            "pidDX",
            "pidDY",
            "closingRate",
            "overshootRisk",
            "dt",
            "confidence",
            "maxSpeedRatio",
        ]

        self.assertEqual(names, expected)
        self.assertEqual(len(names), len(FEATURE_COLUMNS))

    def test_cpp_exposes_eight_axis_direction_helper(self):
        cpp_path = REPO_ROOT / "mouse/PidGovernor.cpp"
        source = cpp_path.read_text(encoding="utf-8")
        header = (REPO_ROOT / "mouse/PidGovernor.h").read_text(encoding="utf-8")

        self.assertIn("PidGovernorDirectionCount = 8", header)
        self.assertIn("pidGovernorEightAxisWeights", header)
        self.assertIn("pidGovernorEightAxisWeights(double errorX, double errorY)", source)
        self.assertIn("errorDirectionDownRight", header)

    def test_cpp_exposes_moving_and_still_helper(self):
        cpp_path = REPO_ROOT / "mouse/PidGovernor.cpp"
        source = cpp_path.read_text(encoding="utf-8")
        header = (REPO_ROOT / "mouse/PidGovernor.h").read_text(encoding="utf-8")

        self.assertIn("PidGovernorMotionStateCount = 2", header)
        self.assertIn("pidGovernorMotionStateWeights", header)
        self.assertIn("pidGovernorMotionStateWeights(double targetVx, double targetVy, double targetSize)", source)
        self.assertIn("targetMotionStill", header)
        self.assertIn("targetMotionMoving", header)


if __name__ == "__main__":
    unittest.main()
