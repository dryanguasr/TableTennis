import unittest

import numpy as np

from table_tennis.constants import (
    AIR_DENSITY,
    BALL_MASS,
    BALL_RADIUS,
    DRAG_COEFFICIENT,
    G,
)
from table_tennis.models import InitialConditions
from table_tennis.physics import (
    apply_table_impact,
    flight_acceleration,
    magnus_coefficient,
    simulate,
    table_restitution,
)


class AcePhysicsTests(unittest.TestCase):
    def test_flight_acceleration_matches_published_equation(self):
        velocity = np.array([5000.0, 0.0, 0.0])
        omega = np.array([0.0, 300.0, 0.0])
        coefficient = (
            0.1
            * np.linalg.norm(velocity)
            / (BALL_RADIUS * np.linalg.norm(omega))
            - 0.001
        )
        drag = (
            -0.5
            * DRAG_COEFFICIENT
            * AIR_DENSITY
            * np.pi
            * BALL_RADIUS**2
            * np.linalg.norm(velocity)
            * velocity
        )
        magnus = (
            -coefficient
            * AIR_DENSITY
            * (4.0 / 3.0)
            * np.pi
            * BALL_RADIUS**3
            * np.cross(velocity, omega)
        )
        expected = (
            (drag + magnus) / BALL_MASS
            + np.array([0.0, 0.0, -G])
        )
        np.testing.assert_allclose(
            flight_acceleration(velocity, omega),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_drag_is_quadratic_and_zero_spin_is_stable(self):
        slow = flight_acceleration((2500.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        fast = flight_acceleration((5000.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(fast[0] / slow[0], 4.0, places=12)
        self.assertEqual(magnus_coefficient((5000.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.0)
        self.assertTrue(np.all(np.isfinite(fast)))

    def test_spin_is_constant_during_free_flight(self):
        initial = InitialConditions(
            pos=(-1000.0, -1000.0, 2000.0),
            vel=(1000.0, 0.0, 1000.0),
            omega=(40.0, -80.0, 120.0),
        )
        result = simulate(initial, dt=0.005, t_max=0.1)
        expected = np.repeat(
            np.asarray(initial.omega, dtype=float)[:, None],
            result.omega.shape[1],
            axis=1,
        )
        np.testing.assert_allclose(result.omega, expected, atol=1e-12)
        np.testing.assert_allclose(result.alpha, 0.0, atol=1e-12)

    def test_table_restitution_uses_vertical_impact_speed(self):
        self.assertAlmostEqual(table_restitution(-3000.0), 0.92, places=12)
        self.assertAlmostEqual(table_restitution(3000.0), 0.92, places=12)

    def test_sliding_table_contact_matches_ace_matrices(self):
        velocity = np.array([3200.0, -700.0, -2800.0])
        omega = np.array([35.0, -55.0, 20.0])
        restitution = table_restitution(velocity[2])
        tangent = np.array(
            [
                velocity[0] - BALL_RADIUS * omega[1],
                velocity[1] + BALL_RADIUS * omega[0],
            ]
        )
        alpha = min(
            2.0 / 5.0,
            0.25
            * (1.0 + restitution)
            * abs(velocity[2])
            / np.linalg.norm(tangent),
        )
        cvv = np.diag([1.0 - alpha, 1.0 - alpha, -restitution])
        cvw = np.array(
            [
                [0.0, alpha * BALL_RADIUS, 0.0],
                [-alpha * BALL_RADIUS, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cwv = np.array(
            [
                [0.0, -1.5 * alpha / BALL_RADIUS, 0.0],
                [1.5 * alpha / BALL_RADIUS, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cww = np.diag(
            [1.0 - 1.5 * alpha, 1.0 - 1.5 * alpha, 1.0]
        )
        post_velocity, post_omega = apply_table_impact(velocity, omega)
        np.testing.assert_allclose(post_velocity, cvv @ velocity + cvw @ omega)
        np.testing.assert_allclose(post_omega, cwv @ velocity + cww @ omega)

    def test_rolling_table_contact_uses_two_fifths_alpha(self):
        velocity = np.array([2000.0, 0.0, -3000.0])
        omega = np.array([0.0, 100.0, 0.0])
        post_velocity, post_omega = apply_table_impact(velocity, omega)
        self.assertAlmostEqual(post_velocity[0], velocity[0], places=12)
        self.assertAlmostEqual(post_omega[1], omega[1], places=12)

    def test_rk4_five_millisecond_solution_converges_to_one_millisecond(self):
        initial = InitialConditions(
            pos=(-300.0, 350.0, 1600.0),
            vel=(6000.0, 500.0, 1800.0),
            omega=(0.0, -280.0, 180.0),
        )
        coarse = simulate(initial, dt=0.005, t_max=0.2)
        fine = simulate(initial, dt=0.001, t_max=0.2)
        np.testing.assert_allclose(
            coarse.x[:, -1],
            fine.x[:, -1],
            atol=0.05,
        )
        np.testing.assert_allclose(
            coarse.v[:, -1],
            fine.v[:, -1],
            atol=0.1,
        )

    def test_table_event_is_interpolated_to_the_contact_plane(self):
        result = simulate(
            InitialConditions(
                pos=(500.0, 500.0, 900.0),
                vel=(1000.0, 0.0, -2500.0),
                omega=(0.0, 0.0, 0.0),
            ),
            dt=0.005,
            t_max=0.1,
        )
        bounces = [
            event for event in result.events if event.kind == "table_bounce"
        ]
        self.assertEqual(len(bounces), 1)
        self.assertAlmostEqual(
            bounces[0].point[2],
            760.0 + BALL_RADIUS,
            places=9,
        )
        self.assertNotAlmostEqual(
            bounces[0].time / 0.005,
            round(bounces[0].time / 0.005),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
