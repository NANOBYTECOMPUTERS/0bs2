import queue
import unittest
from pathlib import Path

from training import train_gui

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class FakeProcess:
    def __init__(self, lines, code=0):
        self.stdout = iter(lines)
        self.code = code

    def wait(self):
        return self.code


class FakeVar:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class FakeLabel:
    def __init__(self):
        self.kwargs = {}

    def config(self, **kwargs):
        self.kwargs.update(kwargs)


class FakeText:
    def __init__(self, initial=""):
        self.text = initial
        self.configs = []

    def configure(self, **kwargs):
        self.configs.append(kwargs)

    def insert(self, index, text):
        self.text += text

    def delete(self, start, end):
        self.text = ""

    def get(self, start, end):
        return self.text

    def see(self, index):
        pass


class TrainGuiLogContractTest(unittest.TestCase):
    def make_app_without_tk(self):
        app = train_gui.TrainingGUI.__new__(train_gui.TrainingGUI)
        app.log_queue = queue.Queue()
        return app

    def test_standard_stream_thread_only_enqueues_main_thread_events(self):
        app = self.make_app_without_tk()
        proc = FakeProcess(["epoch=001 train_loss=1.0\n"], code=0)

        train_gui.TrainingGUI._stream_output(app, proc)

        items = []
        while not app.log_queue.empty():
            items.append(app.log_queue.get_nowait())

        self.assertIn("epoch=001 train_loss=1.0\n", items)
        self.assertIn("\n[Process finished with exit code 0]\n", items)
        self.assertIn((train_gui.LOG_EVENT_PROCESS_FINISHED, 0), items)

    def test_realworld_stream_thread_only_enqueues_main_thread_events(self):
        app = self.make_app_without_tk()
        proc = FakeProcess(["converted samples\n"], code=0)

        train_gui.TrainingGUI._stream_rw_output(app, proc, chain_next=True)

        items = []
        while not app.log_queue.empty():
            items.append(app.log_queue.get_nowait())

        self.assertIn("converted samples\n", items)
        self.assertIn("\n[Process finished with exit code 0]\n", items)
        self.assertIn((train_gui.LOG_EVENT_REALWORLD_FINISHED, 0, True), items)

    def test_epoch_progress_uses_one_based_training_logs_directly(self):
        app = self.make_app_without_tk()
        app.current_epoch = 0
        app.current_max_epochs = 40
        app.progress_var = FakeVar()
        app.progress_label = FakeLabel()

        train_gui.TrainingGUI._parse_epoch_progress(app, "epoch=001 train_loss=1.0\n")

        self.assertEqual(app.current_epoch, 1)
        self.assertAlmostEqual(app.progress_var.value, 2.5)
        self.assertIn("Epoch 1/40", app.progress_label.kwargs["text"])

    def test_append_and_clear_log_mirror_embedded_and_dialog_widgets(self):
        app = self.make_app_without_tk()
        app.log_text = FakeText()
        app.log_dialog_text = FakeText()
        app.current_epoch = 0
        app.current_max_epochs = 0
        app.progress_label = FakeLabel()

        train_gui.TrainingGUI._append_log(app, "epoch=001 visible log\n")

        self.assertEqual(app.log_text.text, "epoch=001 visible log\n")
        self.assertEqual(app.log_dialog_text.text, "epoch=001 visible log\n")

        train_gui.TrainingGUI._clear_log(app)

        self.assertEqual(app.log_text.text, "")
        self.assertEqual(app.log_dialog_text.text, "")

    def test_training_launch_paths_raise_the_log_dialog(self):
        source = (REPO_ROOT / "training" / "train_gui.py").read_text(encoding="utf-8")

        self.assertIn("def _show_training_log_dialog", source)
        self.assertIn("def _hide_training_log_dialog", source)
        self.assertIn("self.log_dialog_text", source)
        self.assertIn("self._show_training_log_dialog()", source)
        self.assertIn("Show Training Log", source)

    def test_gui_defaults_match_current_recommended_training_settings(self):
        tracker = train_gui.DEFAULTS["neural_tracker"]
        self.assertEqual(tracker["epochs"], 30)
        self.assertEqual(tracker["patience"], 8)
        self.assertAlmostEqual(tracker["learning_rate"], 0.00045)
        self.assertAlmostEqual(tracker["weight_decay"], 0.00012)
        self.assertAlmostEqual(tracker["distill_weight"], 0.20)
        self.assertAlmostEqual(tracker["ranking_loss_weight"], 0.20)

        governor = train_gui.DEFAULTS["pid_governor"]
        self.assertTrue(governor["use_gbm_teacher"])
        self.assertEqual(governor["gbm_boosting_rounds"], 300)
        self.assertAlmostEqual(governor["distill_weight"], 0.55)

        temporal = train_gui.DEFAULTS["temporal"]
        self.assertEqual(temporal["dataset"], "training/datasets/temporal_tracks_v2.npz")
        self.assertEqual(temporal["epochs"], 60)
        self.assertEqual(temporal["patience"], 10)
        self.assertEqual(temporal["auto_generate_samples"], 65536)
        self.assertAlmostEqual(temporal["smoothness_weight"], 0.008)
        self.assertAlmostEqual(temporal["velocity_weight"], 0.10)
        self.assertEqual(temporal["near_horizon_frames"], 8)
        self.assertAlmostEqual(temporal["near_horizon_weight"], 3.5)
        self.assertEqual(temporal["latency_focus_start_frame"], 3)
        self.assertEqual(temporal["latency_focus_end_frame"], 6)
        self.assertAlmostEqual(temporal["latency_focus_weight"], 0.75)
        self.assertAlmostEqual(temporal["max_speed_px_s"], 1800.0)


if __name__ == "__main__":
    unittest.main()
