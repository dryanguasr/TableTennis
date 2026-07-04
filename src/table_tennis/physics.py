"""Numerical simulation and ball-contact models.

This module deliberately has no plotting, notebook, CLI, or SciPy dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import (
    BALL_MASS,
    BALL_RADIUS,
    BALL_ROT_INERTIA,
    DRAG,
    DT,
    G,
    MAGNUS,
    NET_EXTRA,
    NET_HEIGHT,
    NET_RESTITUTION,
    ROT_DRAG,
    TABLE_FRICTION,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_RESTITUTION,
    TABLE_WIDTH,
    T_MAX,
)
from .models import (
    InitialConditions,
    RacketImpactParameters,
    SimulationEvent,
    SimulationResult,
)


def simulate(
    ic: InitialConditions,
    dt: float = DT,
    t_max: float = T_MAX,
) -> SimulationResult:
    """Simulate a ball trajectory and emit bounce, net-cross, and net-contact events."""

    t = np.arange(0, t_max + dt, dt)
    n = len(t)
    x = np.zeros((3, n))
    v = np.zeros((3, n))
    a = np.zeros((3, n))
    theta = np.zeros((3, n))
    omega = np.zeros((3, n))
    alpha = np.zeros((3, n))
    events = []
    net_contact_active = False

    x[:, 0] = ic.pos
    v[:, 0] = ic.vel
    omega[:, 0] = ic.omega

    for k in range(1, n):
        force = (
            G * BALL_MASS * np.array([0.0, 0.0, -1.0])
            - DRAG * v[:, k - 1]
            + MAGNUS * np.cross(omega[:, k - 1], v[:, k - 1])
        )
        a[:, k] = force / BALL_MASS
        v[:, k] = v[:, k - 1] + a[:, k] * dt
        x[:, k] = x[:, k - 1] + v[:, k] * dt

        previous_net_offset = x[0, k - 1] - TABLE_LENGTH / 2
        current_net_offset = x[0, k] - TABLE_LENGTH / 2
        if (
            previous_net_offset * current_net_offset <= 0
            and previous_net_offset != current_net_offset
        ):
            fraction = float(
                np.clip(
                    -previous_net_offset
                    / (current_net_offset - previous_net_offset),
                    0.0,
                    1.0,
                )
            )
            crossing_point = x[:, k - 1] + fraction * (x[:, k] - x[:, k - 1])
            events.append(
                SimulationEvent(
                    kind="net_cross",
                    index=k,
                    time=float(t[k - 1] + fraction * dt),
                    point=tuple(float(value) for value in crossing_point),
                )
            )

        torque = -ROT_DRAG * omega[:, k - 1]
        alpha[:, k] = torque / BALL_ROT_INERTIA
        omega[:, k] = omega[:, k - 1] + alpha[:, k] * dt
        theta[:, k] = theta[:, k - 1] + omega[:, k] * dt

        if (
            0 < x[0, k] < TABLE_LENGTH
            and 0 < x[1, k] < TABLE_WIDTH
            and x[2, k] < TABLE_HEIGHT + BALL_RADIUS
        ):
            x[2, k] = TABLE_HEIGHT + BALL_RADIUS
            v[:, k], omega[:, k] = apply_table_impact(v[:, k], omega[:, k])
            events.append(
                SimulationEvent(
                    kind="table_bounce",
                    index=k,
                    time=float(t[k]),
                    point=tuple(float(value) for value in x[:, k]),
                    side="server" if x[0, k] < TABLE_LENGTH / 2 else "receiver",
                )
            )

        in_net = (
            TABLE_LENGTH / 2 - BALL_RADIUS
            <= x[0, k]
            <= TABLE_LENGTH / 2 + BALL_RADIUS
            and -NET_EXTRA <= x[1, k] <= TABLE_WIDTH + NET_EXTRA
            and TABLE_HEIGHT + BALL_RADIUS
            < x[2, k]
            < TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
        )
        if in_net:
            if not net_contact_active:
                events.append(
                    SimulationEvent(
                        kind="net_contact",
                        index=k,
                        time=float(t[k]),
                        point=tuple(float(value) for value in x[:, k]),
                    )
                )
            omega[:, k] = NET_RESTITUTION * omega[:, k]
            v[0, k] = -NET_RESTITUTION * v[0, k]
        net_contact_active = in_net

    return SimulationResult(x, v, a, theta, omega, alpha, t, tuple(events))


def rotation_matrix_xyz(
    angles_deg: Tuple[float, float, float],
) -> np.ndarray:
    """Return the XYZ Euler rotation matrix for angles expressed in degrees."""

    ax, ay, az = np.deg2rad(angles_deg)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


def racket_normal(
    racket_angle: Tuple[float, float, float],
) -> np.ndarray:
    """Compute the unit normal of the racket face from its angle."""

    normal = rotation_matrix_xyz(racket_angle) @ np.array([1.0, 0.0, 0.0])
    return normal / np.linalg.norm(normal)


def _limited_tangential_impulse(
    slip_velocity: np.ndarray,
    normal_impulse_magnitude: float,
    friction: float,
) -> np.ndarray:
    """Return the sticking impulse capped by Coulomb friction."""

    effective_inverse_mass = 1.0 / BALL_MASS + BALL_RADIUS**2 / BALL_ROT_INERTIA
    sticking_impulse = -np.asarray(slip_velocity, dtype=float) / effective_inverse_mass
    maximum_impulse = max(0.0, friction) * abs(normal_impulse_magnitude)
    magnitude = float(np.linalg.norm(sticking_impulse))
    if magnitude <= maximum_impulse or magnitude < 1e-12:
        return sticking_impulse
    return sticking_impulse * (maximum_impulse / magnitude)


def apply_table_impact(
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    restitution: float = TABLE_RESTITUTION,
    friction: float = TABLE_FRICTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a table bounce using normal restitution and Coulomb friction."""

    post_v = np.asarray(velocity, dtype=float).copy()
    post_w = np.asarray(angular_velocity, dtype=float).copy()
    incoming_vz = float(post_v[2])
    normal_impulse_magnitude = BALL_MASS * (1.0 + restitution) * abs(incoming_vz)
    contact_radius = np.array([0.0, 0.0, -BALL_RADIUS])
    slip_velocity = np.array([post_v[0], post_v[1], 0.0]) + np.cross(
        post_w, contact_radius
    )
    tangential_impulse = _limited_tangential_impulse(
        slip_velocity,
        normal_impulse_magnitude,
        friction,
    )
    post_v += tangential_impulse / BALL_MASS
    post_w += np.cross(contact_radius, tangential_impulse) / BALL_ROT_INERTIA
    post_v[2] = -restitution * incoming_vz
    return post_v, post_w


def apply_racket_impact(params: RacketImpactParameters) -> InitialConditions:
    """Compute post-impact initial conditions for a racket strike."""

    ball_v = np.array(params.ball_velocity, dtype=float)
    ball_w = np.array(params.ball_omega, dtype=float)
    racket_v = np.array(params.racket_velocity, dtype=float)
    normal = racket_normal(params.racket_angle)

    relative_v = ball_v - racket_v
    normal_speed = float(np.dot(relative_v, normal))
    restitution = np.clip(params.rubber_restitution, 0.0, 1.5)
    friction = np.clip(params.rubber_friction, 0.0, 2.0)
    if params.contact_model == "legacy":
        relative_normal = normal_speed * normal
        relative_tangent = relative_v - relative_normal
        post_relative_v = (
            relative_tangent * max(0.0, 1.0 - friction)
            - restitution * relative_normal
        )
        post_v = racket_v + post_relative_v
        contact_radius = -BALL_RADIUS * normal
        surface_slip = relative_tangent + np.cross(ball_w, contact_radius)
        post_w = (
            ball_w
            - friction * np.cross(contact_radius, surface_slip) / (BALL_RADIUS**2)
        )
        return InitialConditions(
            pos=params.ball_position,
            vel=tuple(post_v),
            omega=tuple(post_w),
        )
    if params.contact_model != "coulomb":
        raise ValueError(f"Unknown racket contact model: {params.contact_model}")

    normal_impulse = -BALL_MASS * (1.0 + restitution) * normal_speed * normal
    contact_radius = -BALL_RADIUS * normal
    relative_tangent = relative_v - normal_speed * normal
    surface_slip = relative_tangent + np.cross(ball_w, contact_radius)
    tangential_impulse = _limited_tangential_impulse(
        surface_slip,
        float(np.linalg.norm(normal_impulse)),
        friction,
    )
    post_v = ball_v + (normal_impulse + tangential_impulse) / BALL_MASS
    post_w = ball_w + np.cross(contact_radius, tangential_impulse) / BALL_ROT_INERTIA

    return InitialConditions(
        pos=params.ball_position,
        vel=tuple(post_v),
        omega=tuple(post_w),
    )


def simulate_racket_impact(
    params: RacketImpactParameters,
    dt: float = DT,
    t_max: float = T_MAX,
) -> SimulationResult:
    """Apply a racket impact and simulate the resulting trajectory."""

    return simulate(apply_racket_impact(params), dt=dt, t_max=t_max)
