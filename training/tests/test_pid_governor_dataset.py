import unittest

from training.pid_governor.dataset import (
    DatasetConfig,
    DIRECTION_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MOTION_STATE_COLUMNS,
    PidConfig,
    PROFILE_NAMES,
    compute_teacher_labels,
    generate_dataset,
)


class PidGovernorDatasetTests(unittest.TestCase):
    def test_generator_covers_motion_profiles_and_respects_speed_cap(self):
        pid = PidConfig(
            actuator_hz=240,
            kp=0.035,
            ki=0.0025,
            kd=0.0006,
            max_pixel_step=1.2,
            output_scale=0.35,
        )
        cfg = DatasetConfig(episodes_per_profile=1, steps_per_episode=48, seed=7, max_speed_multiple=5.0)

        rows = generate_dataset(pid, cfg)

        profiles = {row["profile"] for row in rows}
        self.assertGreater(len(rows), 200)
        self.assertIn("stopped_center", profiles)
        self.assertIn("stopped_offset", profiles)
        self.assertIn("linear_tracking", profiles)
        self.assertIn("direction_change", profiles)
        self.assertIn("moving_away", profiles)
        self.assertIn("jitter_and_loss", profiles)
        self.assertIn("stop_and_go", profiles)
        self.assertIn("stop_and_go", PROFILE_NAMES)

        max_speed = pid.nominal_speed_px_s * cfg.max_speed_multiple
        observed = max(abs(row["target_speed_x_px_s"]) for row in rows)
        observed = max(observed, max(abs(row["target_speed_y_px_s"]) for row in rows))
        self.assertLessEqual(observed, max_speed + 1e-6)

        for row in rows:
            for column in FEATURE_COLUMNS + LABEL_COLUMNS:
                self.assertIn(column, row)
                self.assertTrue(isinstance(row[column], float))
            self.assertGreaterEqual(row["label_speed_scale"], 0.0)
            self.assertLessEqual(row["label_speed_scale"], 1.0)

    def test_motion_state_features_are_present_and_cover_still_and_moving(self):
        pid = PidConfig()
        cfg = DatasetConfig(episodes_per_profile=1, steps_per_episode=48, seed=19)

        rows = generate_dataset(pid, cfg)

        self.assertEqual(MOTION_STATE_COLUMNS, ["target_motion_still", "target_motion_moving"])
        for column in MOTION_STATE_COLUMNS:
            self.assertIn(column, FEATURE_COLUMNS)

        still_rows = [row for row in rows if row["target_motion_still"] > 0.95]
        moving_rows = [row for row in rows if row["target_motion_moving"] > 0.95]
        self.assertGreater(len(still_rows), 0)
        self.assertGreater(len(moving_rows), 0)

        for row in rows:
            total = row["target_motion_still"] + row["target_motion_moving"]
            self.assertAlmostEqual(total, 1.0, places=6)
            self.assertGreaterEqual(row["target_motion_still"], 0.0)
            self.assertLessEqual(row["target_motion_still"], 1.0)
            self.assertGreaterEqual(row["target_motion_moving"], 0.0)
            self.assertLessEqual(row["target_motion_moving"], 1.0)

    def test_eight_axis_direction_features_are_present_and_normalized(self):
        pid = PidConfig()
        cfg = DatasetConfig(episodes_per_profile=1, steps_per_episode=16, seed=11)

        rows = generate_dataset(pid, cfg)

        self.assertEqual(
            DIRECTION_COLUMNS,
            [
                "error_direction_right",
                "error_direction_down_right",
                "error_direction_down",
                "error_direction_down_left",
                "error_direction_left",
                "error_direction_up_left",
                "error_direction_up",
                "error_direction_up_right",
            ],
        )
        for column in DIRECTION_COLUMNS:
            self.assertIn(column, FEATURE_COLUMNS)

        for row in rows:
            weights = [row[column] for column in DIRECTION_COLUMNS]
            self.assertAlmostEqual(sum(weights), 1.0, places=6)
            for weight in weights:
                self.assertGreaterEqual(weight, 0.0)
                self.assertLessEqual(weight, 1.0)

    def test_teacher_slows_near_small_target_and_boosts_when_lagging(self):
        pid = PidConfig(
            actuator_hz=240,
            kp=0.035,
            ki=0.0025,
            kd=0.0006,
            max_pixel_step=1.2,
            output_scale=0.35,
        )
        cfg = DatasetConfig(max_speed_multiple=5.0)

        near_small = {
            "error_distance_px": 0.20,
            "target_size_px": 10.0,
            "closing_rate_px_s": 30.0,
            "overshoot_risk": 0.0,
            "confidence": 1.0,
            "error_direction_right": 1.0,
            "error_direction_down_right": 0.0,
            "error_direction_down": 0.0,
            "error_direction_down_left": 0.0,
            "error_direction_left": 0.0,
            "error_direction_up_left": 0.0,
            "error_direction_up": 0.0,
            "error_direction_up_right": 0.0,
            "target_motion_still": 1.0,
            "target_motion_moving": 0.0,
        }
        far_lagging = {
            "error_distance_px": 24.0,
            "target_size_px": 10.0,
            "closing_rate_px_s": -180.0,
            "overshoot_risk": 0.0,
            "confidence": 1.0,
            "error_direction_right": 1.0,
            "error_direction_down_right": 0.0,
            "error_direction_down": 0.0,
            "error_direction_down_left": 0.0,
            "error_direction_left": 0.0,
            "error_direction_up_left": 0.0,
            "error_direction_up": 0.0,
            "error_direction_up_right": 0.0,
            "target_motion_still": 0.0,
            "target_motion_moving": 1.0,
        }
        overshoot = {
            "error_distance_px": 4.0,
            "target_size_px": 10.0,
            "closing_rate_px_s": 260.0,
            "overshoot_risk": 1.0,
            "confidence": 1.0,
            "error_direction_right": 0.0,
            "error_direction_down_right": 1.0,
            "error_direction_down": 0.0,
            "error_direction_down_left": 0.0,
            "error_direction_left": 0.0,
            "error_direction_up_left": 0.0,
            "error_direction_up": 0.0,
            "error_direction_up_right": 0.0,
            "target_motion_still": 0.0,
            "target_motion_moving": 1.0,
        }

        near_labels = compute_teacher_labels(near_small, pid, cfg)
        far_labels = compute_teacher_labels(far_lagging, pid, cfg)
        overshoot_labels = compute_teacher_labels(overshoot, pid, cfg)

        self.assertLess(near_labels["label_speed_scale"], far_labels["label_speed_scale"])
        self.assertLess(near_labels["label_kp_scale"], far_labels["label_kp_scale"])
        self.assertLess(near_labels["label_ki_scale"], 0.20)
        self.assertLess(overshoot_labels["label_speed_scale"], far_labels["label_speed_scale"])
        self.assertGreater(overshoot_labels["label_kd_scale"], near_labels["label_kd_scale"])

    def test_teacher_adds_mild_extra_braking_for_diagonal_approach(self):
        pid = PidConfig()
        cfg = DatasetConfig(max_speed_multiple=5.0)
        base = {
            "error_distance_px": 18.0,
            "target_size_px": 18.0,
            "closing_rate_px_s": 80.0,
            "overshoot_risk": 0.0,
            "confidence": 1.0,
            "error_direction_right": 1.0,
            "error_direction_down_right": 0.0,
            "error_direction_down": 0.0,
            "error_direction_down_left": 0.0,
            "error_direction_left": 0.0,
            "error_direction_up_left": 0.0,
            "error_direction_up": 0.0,
            "error_direction_up_right": 0.0,
            "target_motion_still": 1.0,
            "target_motion_moving": 0.0,
        }
        diagonal = dict(base)
        diagonal.update({
            "error_direction_right": 0.0,
            "error_direction_down_right": 1.0,
        })

        horizontal_labels = compute_teacher_labels(base, pid, cfg)
        diagonal_labels = compute_teacher_labels(diagonal, pid, cfg)

        self.assertLess(diagonal_labels["label_speed_scale"], horizontal_labels["label_speed_scale"])
        self.assertGreater(diagonal_labels["label_kd_scale"], horizontal_labels["label_kd_scale"])


if __name__ == "__main__":
    unittest.main()
