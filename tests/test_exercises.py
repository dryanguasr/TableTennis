import pickle
import unittest

import numpy as np

from table_tennis.presets.exercises import (
    EXERCISE_NAMES,
    build_exercise,
    build_exercises,
)
from table_tennis.rally import simulate_exercise
from table_tennis.search.exercises import ExerciseSearchJob
from table_tennis.physics import rotation_matrix_xyz
from table_tennis.visualization.animation import racket_gesture_path
from table_tennis.visualization.exercise_videos import (
    PLAYBACK_SLOWDOWN,
    PRE_ROLL_SECONDS,
    _contact_times,
    _racket_pose,
    build_racket_motion_tracks,
    build_exercise_video_jobs,
    standby_pose,
)


class ExerciseSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            definition.name: simulate_exercise(definition)
            for definition in build_exercises()
        }

    def test_catalog_has_ten_complete_exercises(self):
        self.assertEqual(len(EXERCISE_NAMES), 10)
        self.assertEqual(len(self.results["drive_to_drive"].segments), 10)
        self.assertEqual(
            len(self.results["backhand_to_backhand"].segments),
            10,
        )
        self.assertEqual(len(self.results["figure_eight"].segments), 12)
        self.assertEqual(len(self.results["falkenberg"].segments), 18)
        self.assertEqual(
            len(self.results["third_ball_change_of_pace"].segments),
            13,
        )

    def test_every_exercise_and_segment_is_valid(self):
        for name, result in self.results.items():
            self.assertTrue(result.passed, msg=f"{name}: {result.violations}")
            for segment in result.segments:
                self.assertTrue(
                    segment.validation.passed,
                    msg=f"{name}/{segment.stroke.label}: "
                    f"{segment.validation.violations}",
                )
                self.assertGreaterEqual(
                    segment.validation.net_clearance_mm,
                    5.0,
                )

    def test_segments_preserve_position_velocity_and_spin(self):
        for result in self.results.values():
            for current, following in zip(
                result.segments,
                result.segments[1:],
            ):
                index = current.stop_index
                np.testing.assert_allclose(
                    current.result.x[:, index],
                    following.params.ball_position,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    current.result.v[:, index],
                    following.params.ball_velocity,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    current.result.omega[:, index],
                    following.params.ball_omega,
                    atol=1e-6,
                )

    def test_observed_errors_remain_inside_calibrated_limits(self):
        segments = [
            segment
            for result in self.results.values()
            for segment in result.segments
        ]
        for segment in segments:
            self.assertLessEqual(
                segment.validation.bounce_error_mm,
                segment.stroke.bounce_tolerance_mm,
            )
        self.assertLessEqual(
            max(segment.validation.spin_error_rps for segment in segments),
            35.0,
        )

    def test_patterns_and_role_changes_are_explicit(self):
        eight = build_exercise("figure_eight").strokes[:4]
        self.assertEqual(
            [stroke.label for stroke in eight],
            [
                "A paralela de drive",
                "B cruzada de revés",
                "A paralela de revés",
                "B cruzada de drive",
            ],
        )
        falkenberg = build_exercise("falkenberg").strokes[:6]
        self.assertEqual(
            [stroke.label for stroke in falkenberg[::2]],
            ["Revés", "Pivote de derecha", "Derecha"],
        )
        changes = [
            stroke.hitter
            for stroke in build_exercise(
                "third_ball_change_of_pace"
            ).strokes
            if stroke.label.startswith("Corte defensivo")
        ]
        self.assertEqual(changes, ["near", "far", "near"])

        for name, wing in (
            ("third_ball_change_of_pace", "forehand"),
            ("third_ball_change_of_pace_backhand", "backhand"),
        ):
            result = self.results[name]
            for segment in result.segments:
                self.assertEqual(segment.stroke.target_wing, wing)
                bounce_y = segment.validation.first_bounce[1]
                receiver_is_near = segment.stroke.hitter == "far"
                expected_low_y = (
                    wing == "forehand"
                    if receiver_is_near
                    else wing == "backhand"
                )
                if expected_low_y:
                    self.assertLess(bounce_y, 1525.0 / 2.0)
                else:
                    self.assertGreater(bounce_y, 1525.0 / 2.0)

    def test_strong_and_controlled_topspins_have_distinct_speed(self):
        result = self.results["hit_and_pass_forehand"]
        speeds = {
            segment.stroke.pace: segment.validation.outgoing_speed_mm_s
            for segment in result.segments
            if segment.stroke.kind == "topspin"
        }
        self.assertGreater(speeds["strong"], speeds["controlled"])

    def test_racket_gesture_is_six_hundred_millimetres_long(self):
        path = racket_gesture_path((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(
            float(np.linalg.norm(path[-1] - path[0])),
            600.0,
            places=6,
        )

    def test_standby_pose_is_behind_backhand_with_handle_back(self):
        near_center, near_angle = standby_pose("near")
        far_center, far_angle = standby_pose("far")
        self.assertLess(near_center[0], 0.0)
        self.assertGreater(far_center[0], 2740.0)
        self.assertGreater(near_center[1], 1525.0 / 2.0)
        self.assertLess(far_center[1], 1525.0 / 2.0)
        local_handle = np.array([0.0, -1.0, 0.0])
        near_handle = rotation_matrix_xyz(near_angle) @ local_handle
        far_handle = rotation_matrix_xyz(far_angle) @ local_handle
        self.assertLess(near_handle[0], -0.99)
        self.assertGreater(far_handle[0], 0.99)

    def test_handle_points_left_on_forehand_and_right_on_backhand(self):
        local_handle = np.array([0.0, -1.0, 0.0])
        for exercise, wing in (
            ("drive_to_drive", "forehand"),
            ("backhand_to_backhand", "backhand"),
        ):
            result = self.results[exercise]
            tracks = build_racket_motion_tracks(
                result,
                _contact_times(result),
            )
            for player, player_tracks in tracks.items():
                expected_y = (
                    1.0 if player == "near" else -1.0
                )
                if wing == "backhand":
                    expected_y *= -1.0
                for track in player_tracks:
                    handle = (
                        rotation_matrix_xyz(track.angles[2])
                        @ local_handle
                    )
                    self.assertGreater(handle[1] * expected_y, 0.8)

    def test_bezier_tracks_are_continuous_and_cross_impact(self):
        result = self.results["drive_to_drive"]
        contacts = _contact_times(result)
        tracks = build_racket_motion_tracks(result, contacts)
        for player, player_tracks in tracks.items():
            previous_end = float("-inf")
            for track in player_tracks:
                self.assertGreaterEqual(track.times[0], previous_end)
                previous_end = track.times[-1]
                impact_time = track.times[2]
                center, angle = _racket_pose(
                    player,
                    impact_time,
                    tracks,
                )
                segment = result.segments[track.stroke_index]
                np.testing.assert_allclose(
                    center,
                    segment.params.ball_position,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    angle,
                    track.angles[2],
                    atol=1e-6,
                )
                before, _ = _racket_pose(
                    player,
                    impact_time - 1e-5,
                    tracks,
                )
                after, _ = _racket_pose(
                    player,
                    impact_time + 1e-5,
                    tracks,
                )
                self.assertLess(
                    float(np.linalg.norm(np.asarray(after) - before)),
                    1.0,
                )

        simulation_step = 1.0 / (30.0 * PLAYBACK_SLOWDOWN)
        times = np.arange(
            -PRE_ROLL_SECONDS,
            contacts[-1] + 0.6,
            simulation_step,
        )
        for player in ("near", "far"):
            positions = np.asarray(
                [_racket_pose(player, time, tracks)[0] for time in times]
            )
            frame_distances = np.linalg.norm(
                np.diff(positions, axis=0),
                axis=1,
            )
            self.assertLess(float(np.max(frame_distances)), 180.0)

    def test_search_and_video_jobs_are_pickleable(self):
        pickle.dumps(ExerciseSearchJob("drive_to_drive", 3))
        jobs = build_exercise_video_jobs(
            cycles=3,
            fps=30,
            ffmpeg_path="ffmpeg",
        )
        self.assertEqual(len(jobs), 10)
        pickle.dumps(jobs)


if __name__ == "__main__":
    unittest.main()
