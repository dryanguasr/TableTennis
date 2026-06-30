"""3D table tennis ball simulation.

The module contains a small physics engine that approximates the motion of a
table tennis ball.  It can be executed as a script to visualise the motion or it
can be imported by other tools (e.g. notebooks) to reuse the ``simulate``
function.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Constants in the same units as the MATLAB code (mm, g, mN)
BALL_MASS = 2.7  # g
BALL_RADIUS = 20.25  # mm
BALL_ROT_INERTIA = 2 / 3 * BALL_MASS * BALL_RADIUS**2
TABLE_RESTITUTION = 0.89
NET_RESTITUTION = 0.5
DRAG = 2.7  # mN/(mm/s)
ROT_DRAG = 350.0  # mN mm/[rad/s]
MAGNUS = 0.01  # mN/(mm/s^2)
TABLE_FRICTION = 0.25
TABLE_LENGTH = 2740  # mm
TABLE_WIDTH = 1525  # mm
TABLE_HEIGHT = 760  # mm
NET_HEIGHT = 152.5  # mm
NET_EXTRA = 180  # mm
G = 9800  # mm/s^2

# Simulation parameters
ANIMATE = True
PLOT_PERIOD = 5
YAW = -45
PITCH = 23.5
DT = 0.005
T_MAX = 1.5


@dataclass
class InitialConditions:
    pos: Tuple[float, float, float] = (0.0, TABLE_WIDTH * 4 / 8, TABLE_HEIGHT + 2 * NET_HEIGHT)
    vel: Tuple[float, float, float] = (7000.0, -3000.0, -3000.0)
    omega: Tuple[float, float, float] = (0.0, 0.0, 75.0 * 2 * np.pi)


@dataclass
class SimulationResult:
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    t: Optional[np.ndarray] = None
    events: Tuple["SimulationEvent", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationEvent:
    """Discrete event emitted by the numerical simulation."""

    kind: str
    index: int
    time: float
    point: Tuple[float, float, float]
    side: Optional[str] = None


@dataclass
class RacketImpactParameters:
    """Input parameters for a ball-racket impact model.

    Units match the rest of the module: millimetres, seconds and radians.
    The racket angle is measured in degrees around X, Y and Z and rotates the
    default racket normal (+X).
    """

    ball_velocity: Tuple[float, float, float] = (-2000.0, 0.0, -1000.0)
    ball_omega: Tuple[float, float, float] = (0.0, 0.0, 75.0 * 2 * np.pi)
    rubber_friction: float = 0.6
    rubber_restitution: float = 0.85
    racket_angle: Tuple[float, float, float] = (0.0, -30.0, 0.0)
    racket_velocity: Tuple[float, float, float] = (2000.0, 0.0, 1000.0)
    ball_position: Tuple[float, float, float] = (200.0, TABLE_WIDTH / 2, TABLE_HEIGHT + 240.0)
    contact_model: str = "coulomb"


@dataclass
class TrajectoryMoment:
    name: str
    index: int
    time: float
    point: Tuple[float, float, float]
    interval: Optional[Tuple[float, float]] = None
    midpoint: Optional[Tuple[float, float, float]] = None


def simulate(ic: InitialConditions, dt: float = DT, t_max: float = T_MAX) -> SimulationResult:
    """Simulate the ball trajectory.

    Parameters
    ----------
    ic:
        Initial conditions.
    dt:
        Time step in seconds.
    t_max:
        Total simulation time in seconds.
    """

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
        F = (
            G * BALL_MASS * np.array([0.0, 0.0, -1.0])
            - DRAG * v[:, k - 1]
            + MAGNUS * np.cross(omega[:, k - 1], v[:, k - 1])
        )
        a[:, k] = F / BALL_MASS
        v[:, k] = v[:, k - 1] + a[:, k] * dt
        x[:, k] = x[:, k - 1] + v[:, k] * dt

        previous_net_offset = x[0, k - 1] - TABLE_LENGTH / 2
        current_net_offset = x[0, k] - TABLE_LENGTH / 2
        if previous_net_offset * current_net_offset <= 0 and previous_net_offset != current_net_offset:
            fraction = float(np.clip(-previous_net_offset / (current_net_offset - previous_net_offset), 0.0, 1.0))
            crossing_point = x[:, k - 1] + fraction * (x[:, k] - x[:, k - 1])
            events.append(
                SimulationEvent(
                    kind="net_cross",
                    index=k,
                    time=float(t[k - 1] + fraction * dt),
                    point=tuple(float(value) for value in crossing_point),
                )
            )

        tau = -ROT_DRAG * omega[:, k - 1]
        alpha[:, k] = tau / BALL_ROT_INERTIA
        omega[:, k] = omega[:, k - 1] + alpha[:, k] * dt
        theta[:, k] = theta[:, k - 1] + omega[:, k] * dt

        if (0 < x[0, k] < TABLE_LENGTH and 0 < x[1, k] < TABLE_WIDTH and x[2, k] < TABLE_HEIGHT + BALL_RADIUS):
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
            TABLE_LENGTH / 2 - BALL_RADIUS <= x[0, k] <= TABLE_LENGTH / 2 + BALL_RADIUS
            and -NET_EXTRA <= x[1, k] <= TABLE_WIDTH + NET_EXTRA
            and TABLE_HEIGHT + BALL_RADIUS < x[2, k] < TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
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


def _rotation_matrix_xyz(angles_deg: Tuple[float, float, float]) -> np.ndarray:
    """Return the XYZ Euler rotation matrix for degrees."""

    ax, ay, az = np.deg2rad(angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def racket_normal(racket_angle: Tuple[float, float, float]) -> np.ndarray:
    """Compute the unit normal of the racket face from its angle."""

    normal = _rotation_matrix_xyz(racket_angle) @ np.array([1.0, 0.0, 0.0])
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
    slip_velocity = np.array([post_v[0], post_v[1], 0.0]) + np.cross(post_w, contact_radius)
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
    """Compute post-impact ball initial conditions for a racket strike.

    The model resolves relative ball/racket velocity into normal and tangent
    components. Restitution controls the normal rebound; friction couples
    tangential slip with spin at the contact patch.
    """

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
        post_relative_v = relative_tangent * max(0.0, 1.0 - friction) - restitution * relative_normal
        post_v = racket_v + post_relative_v
        contact_radius = -BALL_RADIUS * normal
        surface_slip = relative_tangent + np.cross(ball_w, contact_radius)
        post_w = ball_w - friction * np.cross(contact_radius, surface_slip) / (BALL_RADIUS**2)
        return InitialConditions(pos=params.ball_position, vel=tuple(post_v), omega=tuple(post_w))
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

    return InitialConditions(pos=params.ball_position, vel=tuple(post_v), omega=tuple(post_w))


def simulate_racket_impact(params: RacketImpactParameters, dt: float = DT, t_max: float = T_MAX) -> SimulationResult:
    """Apply a racket impact and simulate the resulting trajectory."""

    return simulate(apply_racket_impact(params), dt=dt, t_max=t_max)


def identify_trajectory_moments(result: SimulationResult, table_level: float = TABLE_HEIGHT + BALL_RADIUS) -> Dict[int, TrajectoryMoment]:
    """Identify coaching moments 1-6 along a simulated trajectory.

    Moment 1 is the first table bounce on the opposite half; 2 is the rising
    interval after that bounce (including its midpoint); 3 is the apex; 4 is the
    descending interval above the table; 5 is the second arrival at table level;
    6 is the first sample below table level after moment 5, when present.
    """

    t = result.t if result.t is not None else np.arange(result.x.shape[1]) * DT
    x, z, vz = result.x[0], result.x[2], result.v[2]
    moments: Dict[int, TrajectoryMoment] = {}

    bounce_candidates = np.asarray(
        [
            event.index
            for event in result.events
            if event.kind == "table_bounce" and event.side == "receiver"
        ],
        dtype=int,
    )
    if bounce_candidates.size == 0:
        bounce_candidates = np.where((x >= TABLE_LENGTH / 2) & (np.isclose(z, table_level, atol=1e-6)) & (vz > 0))[0]
    if bounce_candidates.size == 0:
        bounce_candidates = np.where((x >= TABLE_LENGTH / 2) & (z <= table_level) & (vz > 0))[0]
    if bounce_candidates.size == 0:
        return moments

    i1 = int(bounce_candidates[0])
    moments[1] = TrajectoryMoment("primer impacto en el lado contrario", i1, float(t[i1]), tuple(result.x[:, i1]))

    later_bounces = bounce_candidates[bounce_candidates > i1]
    if later_bounces.size:
        i5 = int(later_bounces[0])
    else:
        later_indices = np.arange(i1 + 1, result.x.shape[1])
        arrivals = later_indices[z[later_indices] <= table_level]
        i5 = int(arrivals[0]) if arrivals.size else result.x.shape[1] - 1
    arc = np.arange(i1, i5 + 1)
    apex_i = int(arc[np.argmax(z[arc])])

    rising = np.arange(i1 + 1, apex_i)
    if rising.size:
        r0, r1 = int(rising[0]), int(rising[-1])
        mid_i = int(round((r0 + r1) / 2))
        moments[2] = TrajectoryMoment("fase ascendente", mid_i, float(t[mid_i]), tuple(result.x[:, mid_i]), (float(t[r0]), float(t[r1])), tuple(result.x[:, mid_i]))

    moments[3] = TrajectoryMoment("punto más alto", apex_i, float(t[apex_i]), tuple(result.x[:, apex_i]))

    descending = np.arange(apex_i + 1, i5)
    if descending.size:
        d0, d1 = int(descending[0]), int(descending[-1])
        mid_i = int(round((d0 + d1) / 2))
        moments[4] = TrajectoryMoment("fase descendente sobre la mesa", mid_i, float(t[mid_i]), tuple(result.x[:, mid_i]), (float(t[d0]), float(t[d1])), tuple(result.x[:, mid_i]))

    if i5 < result.x.shape[1] - 1:
        moments[5] = TrajectoryMoment("segunda llegada al nivel de la mesa", i5, float(t[i5]), tuple(result.x[:, i5]))
        post = np.arange(i5 + 1, result.x.shape[1])
        below = post[z[post] < table_level]
        if below.size:
            i6 = int(below[0])
            moments[6] = TrajectoryMoment("bola por debajo del nivel de la mesa", i6, float(t[i6]), tuple(result.x[:, i6]))

    return moments


def _elliptic_cylinder(center, radius_y, radius_z, thickness, rotation, resolution=32):
    """Create an elliptic cylinder mesh with local thickness along X."""

    u = np.linspace(0, 2 * np.pi, resolution)
    xs = np.array([-thickness / 2, thickness / 2])
    X, U = np.meshgrid(xs, u)
    local = np.stack([X, radius_y * np.cos(U), radius_z * np.sin(U)], axis=0).reshape(3, -1)
    world = (rotation @ local).reshape(3, *X.shape) + np.array(center).reshape(3, 1, 1)
    return world[0], world[1], world[2]


def draw_racket(ax, center=(0, 0, 0), angle=(0, 0, 0)) -> None:
    """Draw a standard racket as five elliptic blade layers plus a handle."""

    rotation = _rotation_matrix_xyz(angle)
    layers = [
        ("black", 2.0),
        ("#f0c060", 2.2),
        ("#deb887", 6.0),
        ("#f0c060", 2.2),
        ("red", 2.0),
    ]
    offset = -sum(width for _, width in layers) / 2
    for color, width in layers:
        layer_center = np.array(center) + rotation @ np.array([offset + width / 2, 0.0, 0.0])
        X, Y, Z = _elliptic_cylinder(layer_center, 75.0, 80.0, width, rotation)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.9, edgecolor="none")
        offset += width

    handle_center = np.array(center) + rotation @ np.array([0.0, -120.0, 0.0])
    X, Y, Z = _elliptic_cylinder(handle_center, 25.0, 18.0, 95.0, rotation @ _rotation_matrix_xyz((0.0, 0.0, 90.0)))
    ax.plot_surface(X, Y, Z, color="#8b5a2b", alpha=0.95, edgecolor="none")


def _unit_vector(vector, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def _bezier(points: np.ndarray, u: float) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    return _bezier((1.0 - u) * points[:-1] + u * points[1:], u)


def _cubic_bezier_segment(p0, p3, v0, v3, duration: float, samples: int) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p1 = p0 + np.asarray(v0, dtype=float) * duration / 3.0
    p2 = p3 - np.asarray(v3, dtype=float) * duration / 3.0
    points = np.array([p0, p1, p2, p3])
    return np.array([_bezier(points, u) for u in np.linspace(0.0, 1.0, samples)])


def racket_gesture_path(contact_point, racket_velocity, samples: int = 40) -> np.ndarray:
    """Build a short racket path around impact from the racket velocity."""

    direction = _unit_vector(racket_velocity)
    contact_point = np.asarray(contact_point, dtype=float)
    start = contact_point - 100.0 * direction
    end = contact_point + 100.0 * direction
    contact_velocity = np.asarray(racket_velocity, dtype=float)
    segment_duration = 0.08
    before = _cubic_bezier_segment(start, contact_point, (0, 0, 0), contact_velocity, segment_duration, samples // 2)
    after = _cubic_bezier_segment(contact_point, end, contact_velocity, (0, 0, 0), segment_duration, samples - samples // 2)
    return np.vstack([before[:-1], after])


def pre_impact_ball_path(contact_point, ball_velocity, samples: int = 24, duration: float = 0.12) -> np.ndarray:
    """Build a short incoming ball path ending at the racket contact point."""

    contact_point = np.asarray(contact_point, dtype=float)
    ball_velocity = np.asarray(ball_velocity, dtype=float)
    start = contact_point - _unit_vector(ball_velocity, fallback=(-1.0, 0.0, 0.0)) * np.linalg.norm(ball_velocity) * duration
    return np.linspace(start, contact_point, samples)


def _draw_ball(ax, center, color="white") -> None:
    u, v_ang = np.mgrid[0 : 2 * np.pi : 25j, 0 : np.pi : 13j]
    ball_x = BALL_RADIUS * np.cos(u) * np.sin(v_ang) + center[0]
    ball_y = BALL_RADIUS * np.sin(u) * np.sin(v_ang) + center[1]
    ball_z = BALL_RADIUS * np.cos(v_ang) + center[2]
    ax.plot_surface(ball_x, ball_y, ball_z, color=color, edgecolor="none")


def draw_table(ax) -> None:
    """Draw the table, markings and net without clearing the axes."""

    X, Y = np.meshgrid([0, TABLE_LENGTH], [0, TABLE_WIDTH])
    Z = np.ones_like(X) * TABLE_HEIGHT
    ax.plot_surface(X, Y, Z, color="b", alpha=0.5)

    for xs in ([0, 20], [TABLE_LENGTH - 20, TABLE_LENGTH]):
        X, Y = np.meshgrid(xs, [0, TABLE_WIDTH])
        Z = np.ones_like(X) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(X, Y, Z, color="w", edgecolor="none")
    for ys in ([0, 20], [TABLE_WIDTH - 20, TABLE_WIDTH], [TABLE_WIDTH / 2 - 5, TABLE_WIDTH / 2 + 5]):
        X, Y = np.meshgrid([0, TABLE_LENGTH], ys)
        Z = np.ones_like(X) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(X, Y, Z, color="w", edgecolor="none")

    Y, Z = np.meshgrid([-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30], [TABLE_HEIGHT + 130, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="w")
    Y, Z = np.meshgrid([-NET_EXTRA, -NET_EXTRA + 30], [TABLE_HEIGHT, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="k")
    Y, Z = np.meshgrid([TABLE_WIDTH + NET_EXTRA - 30, TABLE_WIDTH + NET_EXTRA], [TABLE_HEIGHT, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="k")

    net_square_side = 20
    net_lines_v = np.arange(-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30 + net_square_side, net_square_side)
    net_lines_h = np.arange(TABLE_HEIGHT, TABLE_HEIGHT + 130 + net_square_side, net_square_side)
    for z in net_lines_h:
        ax.plot(
            [TABLE_LENGTH / 2, TABLE_LENGTH / 2],
            [-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30],
            [z, z],
            "k",
            linewidth=0.5,
        )
    for y in net_lines_v:
        ax.plot(
            [TABLE_LENGTH / 2, TABLE_LENGTH / 2],
            [y, y],
            [TABLE_HEIGHT, TABLE_HEIGHT + 130],
            "k",
            linewidth=0.5,
        )


def plot_table(
    ax,
    pos,
    vel,
    acc,
    orient,
    ang_vel,
    ang_acc,
    yaw,
    pitch,
    racket_center=None,
    racket_angle=(0.0, -10.0, 0.0),
    racket_path=None,
) -> None:
    """Draw the table, ball and vectors for a single frame."""

    ax.clear()

    draw_table(ax)

    _draw_ball(ax, pos)

    ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], length=10 / 98, color="g")
    ax.quiver(pos[0], pos[1], pos[2], acc[0], acc[1], acc[2], length=1 / 49, color="c")
    ax.quiver(pos[0], pos[1], pos[2], ang_vel[0], ang_vel[1], ang_vel[2], length=1, color="r")
    if racket_path is not None:
        racket_path = np.asarray(racket_path, dtype=float)
        ax.plot(racket_path[:, 0], racket_path[:, 1], racket_path[:, 2], color="black", linestyle=":", alpha=0.65)

    if racket_center is None:
        racket_center = (120.0, TABLE_WIDTH * 5 / 8, TABLE_HEIGHT + 300.0)
    draw_racket(ax, center=racket_center, angle=racket_angle)

    ax.view_init(pitch, yaw)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(-500, TABLE_LENGTH + 500)
    ax.set_ylim(-500, TABLE_WIDTH + 500)
    ax.set_zlim(0, 1500)
    ax.grid(True)


def resolve_ffmpeg_path(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """Return an executable FFmpeg path when MP4 output is available."""

    if ffmpeg_path:
        path = Path(ffmpeg_path)
        if path.exists():
            return str(path)
        resolved = shutil.which(ffmpeg_path)
        if resolved:
            return resolved
        return None

    return shutil.which("ffmpeg")


def animate_simulation(
    result: SimulationResult,
    save: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    racket_path=None,
    racket_angle=(0.0, -10.0, 0.0),
) -> None:
    """Animate the trajectory using ``matplotlib.animation``."""

    import matplotlib.pyplot as plt
    from matplotlib import animation

    ffmpeg = resolve_ffmpeg_path(ffmpeg_path) if save else None
    if save and ffmpeg is None:
        raise RuntimeError(
            "Cannot save MP4 because FFmpeg was not found. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        racket_center = None
        if racket_path is not None:
            path = np.asarray(racket_path, dtype=float)
            path_index = int(round(frame / max(1, result.x.shape[1] - 1) * (len(path) - 1)))
            racket_center = path[min(path_index, len(path) - 1)]

        plot_table(
            ax,
            result.x[:, frame],
            result.v[:, frame],
            result.a[:, frame],
            result.theta[:, frame],
            result.omega[:, frame],
            result.alpha[:, frame],
            YAW,
            PITCH,
            racket_center=racket_center,
            racket_angle=racket_angle,
            racket_path=racket_path,
        )

    frames = range(0, result.x.shape[1], PLOT_PERIOD)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=DT * 1000 * PLOT_PERIOD)

    if save:
        writer = animation.FFMpegWriter(fps=int(1 / (DT * PLOT_PERIOD)))
        try:
            ani.save(save, writer=writer)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cannot save MP4 because Matplotlib could not start FFmpeg. "
                "Install FFmpeg and add it to PATH, or pass its executable path with --ffmpeg."
            ) from exc
    else:
        plt.show()


def animate_racket_impact(
    result: SimulationResult,
    params: RacketImpactParameters,
    save: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
) -> None:
    """Animate a racket-impact simulation with the pre-impact contact timing.

    ``simulate_racket_impact`` returns a trajectory whose first sample is already
    the contact point after the racket strike. The racket must therefore be at
    the middle of its gesture at result frame 0; the first half of the gesture is
    drawn only during synthetic pre-impact frames.
    """

    import matplotlib.pyplot as plt
    from matplotlib import animation

    ffmpeg = resolve_ffmpeg_path(ffmpeg_path) if save else None
    if save and ffmpeg is None:
        raise RuntimeError(
            "Cannot save MP4 because FFmpeg was not found. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg

    racket_path = racket_gesture_path(params.ball_position, params.racket_velocity)
    incoming_ball = pre_impact_ball_path(params.ball_position, params.ball_velocity)
    dt = float(result.t[1] - result.t[0]) if result.t is not None and len(result.t) > 1 else DT
    post_frames = list(range(0, result.x.shape[1], PLOT_PERIOD))
    if post_frames[-1] != result.x.shape[1] - 1:
        post_frames.append(result.x.shape[1] - 1)

    total_frames = len(incoming_ball) + len(post_frames)
    interval_ms = dt * 1000.0 * PLOT_PERIOD
    contact_index = len(racket_path) // 2 - 1

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def set_axes() -> None:
        ax.view_init(PITCH, YAW)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim(-500, TABLE_LENGTH + 500)
        ax.set_ylim(-500, TABLE_WIDTH + 500)
        ax.set_zlim(0, 1500)
        ax.grid(True)

    def update(frame: int):
        ax.clear()
        draw_table(ax)
        ax.plot(racket_path[:, 0], racket_path[:, 1], racket_path[:, 2], color="black", linestyle=":", alpha=0.65)
        ax.plot(incoming_ball[:, 0], incoming_ball[:, 1], incoming_ball[:, 2], color="orange", linestyle="--", alpha=0.7)

        if frame < len(incoming_ball):
            ball_center = incoming_ball[frame]
            racket_index = int(frame / max(1, len(incoming_ball) - 1) * contact_index)
        else:
            post_frame = post_frames[frame - len(incoming_ball)]
            ball_center = result.x[:, post_frame]
            progress = (frame - len(incoming_ball)) / max(1, len(post_frames) - 1)
            racket_index = min(contact_index + 1 + int(progress * (len(racket_path) - contact_index - 1)), len(racket_path) - 1)
            ax.plot(result.x[0, : post_frame + 1], result.x[1, : post_frame + 1], result.x[2, : post_frame + 1], color="purple")
            ax.quiver(ball_center[0], ball_center[1], ball_center[2], result.v[0, post_frame], result.v[1, post_frame], result.v[2, post_frame], length=10 / 98, color="g")
            ax.quiver(ball_center[0], ball_center[1], ball_center[2], result.a[0, post_frame], result.a[1, post_frame], result.a[2, post_frame], length=1 / 49, color="c")
            ax.quiver(ball_center[0], ball_center[1], ball_center[2], result.omega[0, post_frame], result.omega[1, post_frame], result.omega[2, post_frame], length=1, color="r")

        draw_racket(ax, center=racket_path[racket_index], angle=params.racket_angle)
        _draw_ball(ax, ball_center)
        set_axes()

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms)

    if save:
        writer = animation.FFMpegWriter(fps=int(1 / (dt * PLOT_PERIOD)))
        try:
            ani.save(save, writer=writer)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cannot save MP4 because Matplotlib could not start FFmpeg. "
                "Install FFmpeg and add it to PATH, or pass its executable path with --ffmpeg."
            ) from exc
    else:
        plt.show()


def plot_results(result: SimulationResult, dt: float = DT) -> None:
    """Plot position, velocity and rotation graphs after the simulation."""

    import matplotlib.pyplot as plt

    t = np.arange(result.x.shape[1]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, result.x.T / 1000)
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("position [m]")
    axes[0, 0].set_ylim(0, 3)

    axes[1, 0].plot(t, result.v.T / 1000)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("velocity [m/s]")

    axes[0, 1].plot(t, result.theta.T / (2 * np.pi))
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("rotation [rev]")

    axes[1, 1].plot(t, result.omega.T / (2 * np.pi))
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("angular velocity [rev/s]")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="3D table tennis simulation")
    parser.add_argument("--save", metavar="FILE", help="Save the animation to an mp4 file")
    args = parser.parse_args()

    result = simulate(InitialConditions())
    animate_simulation(result, save=args.save)


if __name__ == "__main__":
    main()
