import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class RealWorldFineTuningContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")

    def test_runtime_logging_config_gui_and_loop_are_default_off(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        draw_neural = self.read("overlay/draw_neural.cpp")
        loop_cpp = self.read("runtime/mouse_thread_loop.cpp")
        mouse_h = self.read("mouse/mouse.h")
        box_h = self.read("mouse/BoxTarget.h")

        for token in (
            "log_real_world_data",
            "real_world_data_log_dir",
        ):
            self.assertIn(token, config_h)
            self.assertIn(token, config_cpp)
            self.assertIn(token, draw_neural)

        self.assertIn("log_real_world_data = false", config_cpp)
        self.assertIn('get_bool("log_real_world_data", false)', config_cpp)
        self.assertIn("training/datasets/real_world", config_cpp)
        self.assertIn("Real-world data logging", draw_neural)
        self.assertIn("RealWorldDataLogger", loop_cpp)
        self.assertIn("realWorldLogger.setEnabled", loop_cpp)
        self.assertIn("realWorldLogger.append", loop_cpp)
        self.assertIn("getLastAppliedMouseDelta", mouse_h)
        self.assertIn("trackHistory", box_h)

    def test_converter_outputs_temporal_and_targeting_npz(self):
        from training.data_gen.convert_real_world_logs import convert_real_world_logs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_len = 4
            horizon = 3
            rows = 10
            history = np.zeros((rows, history_len, 8), dtype=np.float32)
            state = np.zeros((rows, 16), dtype=np.float32)
            for i in range(rows):
                x = 100.0 + i * 3.0
                y = 90.0 + i * 2.0
                for h in range(history_len):
                    history[i, h] = np.array(
                        [x - (history_len - h), y - (history_len - h), 24, 40, 180, 120, 1, 0.85],
                        dtype=np.float32,
                    )
                state[i] = np.array(
                    [i / 60.0, 7, x, y, 24, 40, 180, 120, 1, 0.85, 0.2, 0.1, 1.0, 0.5, 320, 1],
                    dtype=np.float32,
                )

            log_path = root / "session.npz"
            np.savez_compressed(log_path, history=history, state=state)

            temporal_out = root / "temporal_real.npz"
            targeting_out = root / "targeting_real.npz"
            summary = convert_real_world_logs(
                inputs=[log_path],
                temporal_output=temporal_out,
                targeting_output=targeting_out,
                history_length=history_len,
                prediction_horizon=horizon,
                synthetic_temporal=None,
                synthetic_targeting=None,
                mix_ratio=1.0,
                heldout_fraction=0.0,
                seed=123,
            )

            self.assertGreater(summary["temporal_samples"], 0)
            self.assertGreater(summary["targeting_samples"], 0)
            with np.load(temporal_out, allow_pickle=False) as temporal:
                self.assertEqual(temporal["history"].shape[-2:], (history_len, 8))
                self.assertEqual(temporal["future"].shape[-2:], (horizon, 2))
            with np.load(targeting_out, allow_pickle=False) as targeting:
                self.assertIn("features", targeting)
                self.assertIn("refinement", targeting)
                self.assertIn("confidence", targeting)

    def test_trainers_and_evaluators_expose_real_world_finetune_flags(self):
        for path in (
            "training/train_temporal.py",
            "training/train_neural_targeting.py",
        ):
            source = self.read(path)
            for token in (
                "--fine-tune",
                "--checkpoint",
                "--real-world-data",
                "--mix-ratio",
                "realworld.onnx",
            ):
                self.assertIn(token, source)

        for path in (
            "training/evaluate.py",
            "training/evaluate_neural_targeting.py",
        ):
            source = self.read(path)
            self.assertIn("--compare-model", source)
            self.assertIn("--real-world-data", source)
            self.assertIn("comparison", source)


if __name__ == "__main__":
    unittest.main()
