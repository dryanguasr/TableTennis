import pickle
import unittest
from dataclasses import replace

import numpy as np

import return_parameter_search as returns
from generate_return_videos import _timeline_indices
from table_tennis_simulation import (
    RacketImpactParameters,
    SimulationEvent,
    apply_racket_impact,
    apply_table_impact,
    identify_trajectory_moments,
    simulate_racket_impact,
)


class ReturnSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service_result = simulate_racket_impact(returns.PILOT_SERVICE_PARAMS, t_max=3.0)

    def test_pilot_service_is_legal(self):
        report = returns.validate_service(
            returns.PILOT_SERVICE_PARAMS,
            self.service_result,
            returns.ServiceTargets(),
        )
        self.assertTrue(report.passed, report.violations)
        self.assertLessEqual(report.bounce_error_mm, 75.0)
        self.assertLessEqual(report.spin_error_rps, 10.0)

    def test_moments_are_inside_first_receiver_arc(self):
        moments = identify_trajectory_moments(self.service_result)
        self.assertTrue(all(number in moments for number in (1, 2, 3, 4, 5)))
        indices = [moments[number].index for number in (1, 2, 3, 4, 5)]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), len(set(indices)))

    def test_bank_has_ten_valid_and_distinct_strokes(self):
        count = 0
        for name in returns.RETURN_PRESET_VECTORS:
            params_by_side = {}
            for side in ("forehand", "backhand"):
                _, _, params = returns.build_return_preset(name, side, self.service_result)
                depth = {
                    "cut_short": "short",
                    "cut_two_bounce": "two_bounce",
                    "cut_long": "long",
                    "top_two_bounce": "two_bounce",
                    "top_long": "long",
                }[name]
                spin = (-15.0, 35.0, 10.0) if name.startswith("cut_") else (0.0, -45.0, 0.0)
                targets = returns.StrokeTargets(depth=depth, spin_rps=spin, stroke_side=side)
                report = returns.validate_return(
                    params,
                    simulate_racket_impact(params, t_max=3.0),
                    targets,
                )
                self.assertTrue(report.passed, (name, side, report.violations))
                params_by_side[side] = params
                count += 1
            self.assertNotEqual(
                params_by_side["forehand"].racket_angle[0],
                params_by_side["backhand"].racket_angle[0],
            )
            self.assertTrue(
                np.allclose(
                    params_by_side["forehand"].racket_velocity,
                    params_by_side["backhand"].racket_velocity,
                )
            )
        self.assertEqual(count, 10)

    def test_coulomb_table_friction_cannot_reverse_a_shallow_fast_ball(self):
        velocity, _ = apply_table_impact(
            np.array([-1800.0, 0.0, -500.0]),
            np.array([0.0, 45.0 * 2 * np.pi, 0.0]),
        )
        self.assertLess(velocity[0], 0.0)

    def test_zero_rubber_friction_preserves_spin_and_tangent(self):
        params = RacketImpactParameters(
            ball_velocity=(1000.0, 250.0, -100.0),
            ball_omega=(10.0, 20.0, 30.0),
            rubber_friction=0.0,
            rubber_restitution=0.8,
            racket_angle=(0.0, 0.0, 0.0),
            racket_velocity=(0.0, 0.0, 0.0),
        )
        post = apply_racket_impact(params)
        self.assertAlmostEqual(post.vel[0], -800.0)
        self.assertAlmostEqual(post.vel[1], 250.0)
        self.assertAlmostEqual(post.vel[2], -100.0)
        self.assertTrue(np.allclose(post.omega, params.ball_omega))

    def test_cut_short_moves_downward_and_keeps_forward_direction(self):
        _, _, params = returns.build_return_preset("cut_short", "forehand", self.service_result)
        result = simulate_racket_impact(params, t_max=3.0)
        bounces = returns.table_bounces(result, "server")
        self.assertLess(params.racket_velocity[0], 0.0)
        self.assertLess(params.racket_velocity[2], 0.0)
        self.assertGreaterEqual(len(bounces), 2)
        self.assertLess(result.v[0, bounces[0].index], 0.0)
        self.assertLess(bounces[1].point[0], bounces[0].point[0])

    def test_video_timeline_reaches_five_seconds_or_floor(self):
        _, contact_index, params = returns.build_return_preset(
            "cut_short",
            "forehand",
            self.service_result,
        )
        return_result = simulate_racket_impact(params, t_max=5.0)
        pre_frames, service_frames, return_frames = _timeline_indices(
            self.service_result,
            contact_index,
            return_result,
            fps=30,
            max_duration=5.0,
        )
        total_frames = pre_frames + len(service_frames) + len(return_frames)
        if total_frames < 150:
            self.assertLess(return_result.x[2, return_frames[-1]], 0.0)
        else:
            self.assertEqual(total_frames, 150)

    def test_validator_rejects_net_contact_and_wrong_target(self):
        _, _, params = returns.build_return_preset("top_two_bounce", "backhand", self.service_result)
        result = simulate_racket_impact(params, t_max=3.0)
        first_bounce = returns.table_bounces(result)[0]
        fake_event = SimulationEvent(
            kind="net_contact",
            index=1,
            time=max(0.0, first_bounce.time - 0.1),
            point=(1370.0, 762.5, 850.0),
        )
        touched = replace(result, events=result.events + (fake_event,))
        targets = returns.StrokeTargets(
            depth="two_bounce",
            direction="elbow",
            spin_rps=(0.0, -45.0, 0.0),
            stroke_side="backhand",
        )
        self.assertFalse(returns.validate_return(params, touched, targets).passed)
        wrong_target = replace(targets, direction="forehand", bounce_tolerance_mm=20.0)
        self.assertFalse(returns.validate_return(params, result, wrong_target).passed)

    def test_objectives_are_pickleable_and_worker_settings_do_not_crash(self):
        targets = returns.StrokeTargets(
            depth="two_bounce",
            spin_rps=(0.0, -45.0, 0.0),
        )
        objective = returns._ReturnObjective(
            self.service_result,
            targets,
            returns.RubberProperties(),
            4,
            0.005,
            1.0,
        )
        pickle.dumps(objective)
        for workers in (1, 4):
            result = returns.search_return(
                self.service_result,
                targets,
                config=returns.SearchConfig(
                    maxiter=0,
                    popsize=1,
                    restarts=1,
                    workers=workers,
                    polish=False,
                    t_max=1.0,
                ),
                use_validated_preset=False,
            )
            self.assertIsNotNone(result.validation)


if __name__ == "__main__":
    unittest.main()
