"""Numerical simulation and ball-contact models.

This module deliberately has no plotting, notebook, CLI, or SciPy dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import (
    AIR_DENSITY,
    BALL_MASS,
    BALL_RADIUS,
    BALL_ROT_INERTIA,
    DRAG_COEFFICIENT,
    DT,
    G,
    MAGNUS_OFFSET,
    MAGNUS_SCALE,
    NET_EXTRA,
    NET_HEIGHT,
    NET_RESTITUTION,
    TABLE_FRICTION,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
    T_MAX,
)
from .models import (
    InitialConditions,
    RacketImpactParameters,
    SimulationEvent,
    SimulationResult,
)


_NORM_EPSILON = 1e-12
_TABLE_IMPACT_BISECTIONS = 12


def magnus_coefficient(
    velocity: np.ndarray | tuple[float, float, float],
    angular_velocity: np.ndarray | tuple[float, float, float],
) -> float:
    """Return the velocity-dependent ACE Magnus coefficient."""

    speed = float(np.linalg.norm(velocity))
    spin = float(np.linalg.norm(angular_velocity))
    if speed <= _NORM_EPSILON or spin <= _NORM_EPSILON:
        return 0.0
    return (
        MAGNUS_SCALE * speed / (BALL_RADIUS * spin)
        - MAGNUS_OFFSET
    )


def flight_acceleration(
    velocity: np.ndarray | tuple[float, float, float],
    angular_velocity: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Evaluate the ACE drag, Magnus, and gravitational acceleration."""

    ball_v = np.asarray(velocity, dtype=float)
    ball_w = np.asarray(angular_velocity, dtype=float)
    speed = float(np.linalg.norm(ball_v))
    drag_force = (
        -0.5
        * DRAG_COEFFICIENT
        * AIR_DENSITY
        * np.pi
        * BALL_RADIUS**2
        * speed
        * ball_v
    )
    coefficient = magnus_coefficient(ball_v, ball_w)
    magnus_force = (
        -coefficient
        * AIR_DENSITY
        * (4.0 / 3.0)
        * np.pi
        * BALL_RADIUS**3
        * np.cross(ball_v, ball_w)
    )
    return (
        (drag_force + magnus_force) / BALL_MASS
        + np.array([0.0, 0.0, -G])
    )


def _flight_acceleration_components(
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
) -> tuple[float, float, float]:
    """Allocation-free acceleration evaluation for the RK4 inner loop."""

    speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
    spin = float(np.sqrt(wx * wx + wy * wy + wz * wz))
    drag_factor = (
        -0.5
        * DRAG_COEFFICIENT
        * AIR_DENSITY
        * np.pi
        * BALL_RADIUS**2
        * speed
        / BALL_MASS
    )
    coefficient = (
        0.0
        if speed <= _NORM_EPSILON or spin <= _NORM_EPSILON
        else MAGNUS_SCALE * speed / (BALL_RADIUS * spin)
        - MAGNUS_OFFSET
    )
    magnus_factor = (
        -coefficient
        * AIR_DENSITY
        * (4.0 / 3.0)
        * np.pi
        * BALL_RADIUS**3
        / BALL_MASS
    )
    cross_x = vy * wz - vz * wy
    cross_y = vz * wx - vx * wz
    cross_z = vx * wy - vy * wx
    return (
        drag_factor * vx + magnus_factor * cross_x,
        drag_factor * vy + magnus_factor * cross_y,
        drag_factor * vz + magnus_factor * cross_z - G,
    )


def _rk4_flight_step(
    position: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a free-flight state using a fixed-step fourth-order RK method."""

    if duration <= 0.0:
        return position.copy(), velocity.copy()

    wx, wy, wz = (
        float(angular_velocity[0]),
        float(angular_velocity[1]),
        float(angular_velocity[2]),
    )
    k1_x = velocity
    k1_v = np.asarray(
        _flight_acceleration_components(*velocity, wx, wy, wz)
    )
    k2_x = velocity + 0.5 * duration * k1_v
    k2_v = np.asarray(
        _flight_acceleration_components(*k2_x, wx, wy, wz)
    )
    k3_x = velocity + 0.5 * duration * k2_v
    k3_v = np.asarray(
        _flight_acceleration_components(*k3_x, wx, wy, wz)
    )
    k4_x = velocity + duration * k3_v
    k4_v = np.asarray(
        _flight_acceleration_components(*k4_x, wx, wy, wz)
    )
    next_position = position + duration * (
        k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
    ) / 6.0
    next_velocity = velocity + duration * (
        k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
    ) / 6.0
    return next_position, next_velocity


def _table_impact_state(
    position: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    duration: float,
    end_position: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Locate a downward table-plane crossing inside one integration step."""

    table_level = TABLE_HEIGHT + BALL_RADIUS
    if position[2] < table_level or end_position[2] >= table_level:
        return None

    low = 0.0
    high = duration
    for _ in range(_TABLE_IMPACT_BISECTIONS):
        midpoint = 0.5 * (low + high)
        midpoint_position, _ = _rk4_flight_step(
            position,
            velocity,
            angular_velocity,
            midpoint,
        )
        if midpoint_position[2] >= table_level:
            low = midpoint
        else:
            high = midpoint
    impact_time = 0.5 * (low + high)
    impact_position, impact_velocity = _rk4_flight_step(
        position,
        velocity,
        angular_velocity,
        impact_time,
    )
    impact_position[2] = table_level
    if not (
        0.0 < impact_position[0] < TABLE_LENGTH
        and 0.0 < impact_position[1] < TABLE_WIDTH
    ):
        return None
    return impact_time, impact_position, impact_velocity


def simulate(
    ic: InitialConditions,
    dt: float = DT,
    t_max: float = T_MAX,
) -> SimulationResult:
    """Simulate a ball using the deterministic ACE baseline equations."""

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
    a[:, 0] = flight_acceleration(v[:, 0], omega[:, 0])

    for k in range(1, n):
        previous_position = x[:, k - 1].copy()
        previous_velocity = v[:, k - 1].copy()
        previous_omega = omega[:, k - 1].copy()
        next_position, next_velocity = _rk4_flight_step(
            previous_position,
            previous_velocity,
            previous_omega,
            dt,
        )
        next_omega = previous_omega.copy()
        theta_increment = previous_omega * dt
        step_events = []

        impact = _table_impact_state(
            previous_position,
            previous_velocity,
            previous_omega,
            dt,
            next_position,
        )
        if impact is not None:
            impact_duration, impact_position, impact_velocity = impact
            post_velocity, post_omega = apply_table_impact(
                impact_velocity,
                previous_omega,
            )
            remaining = dt - impact_duration
            next_position, next_velocity = _rk4_flight_step(
                impact_position,
                post_velocity,
                post_omega,
                remaining,
            )
            next_omega = post_omega
            theta_increment = (
                previous_omega * impact_duration
                + post_omega * remaining
            )
            step_events.append(
                SimulationEvent(
                    kind="table_bounce",
                    index=k,
                    time=float(t[k - 1] + impact_duration),
                    point=tuple(float(value) for value in impact_position),
                    side=(
                        "server"
                        if impact_position[0] < TABLE_LENGTH / 2
                        else "receiver"
                    ),
                )
            )

        x[:, k] = next_position
        v[:, k] = next_velocity
        omega[:, k] = next_omega
        theta[:, k] = theta[:, k - 1] + theta_increment

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
            step_events.append(
                SimulationEvent(
                    kind="net_cross",
                    index=k,
                    time=float(t[k - 1] + fraction * dt),
                    point=tuple(float(value) for value in crossing_point),
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
                step_events.append(
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
        a[:, k] = flight_acceleration(v[:, k], omega[:, k])
        events.extend(sorted(step_events, key=lambda event: event.time))

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
    restitution: float | None = None,
    friction: float = TABLE_FRICTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an instantaneous ACE ball-table contact."""

    incoming_v = np.asarray(velocity, dtype=float)
    incoming_w = np.asarray(angular_velocity, dtype=float)
    incoming_vz = float(incoming_v[2])
    resolved_restitution = (
        table_restitution(incoming_vz)
        if restitution is None
        else float(restitution)
    )
    tangent_velocity = np.array(
        [
            incoming_v[0] - BALL_RADIUS * incoming_w[1],
            incoming_v[1] + BALL_RADIUS * incoming_w[0],
        ]
    )
    tangent_speed = float(np.linalg.norm(tangent_velocity))
    if tangent_speed <= _NORM_EPSILON:
        contact_alpha = 2.0 / 5.0
    else:
        contact_alpha = min(
            2.0 / 5.0,
            max(0.0, friction)
            * (1.0 + resolved_restitution)
            * abs(incoming_vz)
            / tangent_speed,
        )

    post_v = np.array(
        [
            (1.0 - contact_alpha) * incoming_v[0]
            + contact_alpha * BALL_RADIUS * incoming_w[1],
            (1.0 - contact_alpha) * incoming_v[1]
            - contact_alpha * BALL_RADIUS * incoming_w[0],
            -resolved_restitution * incoming_vz,
        ]
    )
    spin_scale = 1.0 - 1.5 * contact_alpha
    post_w = np.array(
        [
            spin_scale * incoming_w[0]
            - 1.5 * contact_alpha * incoming_v[1] / BALL_RADIUS,
            spin_scale * incoming_w[1]
            + 1.5 * contact_alpha * incoming_v[0] / BALL_RADIUS,
            incoming_w[2],
        ]
    )
    return post_v, post_w


def table_restitution(vertical_velocity: float) -> float:
    """Return the ACE table restitution for an incoming vertical speed."""

    speed_metres_per_second = abs(float(vertical_velocity)) / 1000.0
    return float(np.clip(0.98 - 0.02 * speed_metres_per_second, 0.0, 1.0))


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
