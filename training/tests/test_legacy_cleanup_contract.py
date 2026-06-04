import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


LEGACY_KEYS = (
    "minSpeedMultiplier",
    "maxSpeedMultiplier",
    "predictionInterval",
    "snapRadius",
    "nearRadius",
    "speedCurveExponent",
    "snapBoostFactor",
)


class LegacyCleanupContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_legacy_prediction_keys_remain_config_compatible(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")

        self.assertIn("Legacy prediction compatibility", config_h)
        self.assertIn("loadable for old configs", config_h)
        for key in LEGACY_KEYS:
            self.assertIn(key, config_h)
            self.assertIn(f'get_double("{key}"', config_cpp)
            self.assertIn(f'MERGE_FIELD("{key}", {key})', config_cpp)

    def test_active_legacy_use_sites_are_marked_as_retained_compatibility(self):
        overlay_loop = self.read("runtime/game_overlay_loop.cpp")

        self.assertIn("Legacy prediction compatibility", overlay_loop)
        self.assertIn("retained for aim-sim and overlay demo parity", overlay_loop)
        self.assertIn("config.predictionInterval", overlay_loop)
        for key in (
            "config.minSpeedMultiplier",
            "config.maxSpeedMultiplier",
            "config.snapRadius",
            "config.nearRadius",
            "config.speedCurveExponent",
            "config.snapBoostFactor",
        ):
            self.assertIn(key, overlay_loop)

    def test_cleanup_inventory_documents_deprecate_vs_remove_decision(self):
        readme = self.read("README.md")

        self.assertIn("Legacy Prediction Compatibility", readme)
        self.assertIn("Deprecated but retained", readme)
        self.assertIn("Removal gate", readme)
        self.assertIn("disabled-neural behavior stays identical", readme)
        for key in LEGACY_KEYS + ("kalman_warmup_frames", "kalman_acquisition_frames"):
            self.assertIn(key, readme)


if __name__ == "__main__":
    unittest.main()
