"""Physically chained, bidirectional multi-stroke table-tennis rallies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import zlib

import numpy as np

from .constants import (
    BALL_RADIUS,
    NET_HEIGHT,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from .events import net_crossings, table_bounces
from .models import InitialConditions, RacketImpactParameters, SimulationResult
from .physics import apply_racket_impact, racket_normal, simulate_racket_impact


Player = Literal["near", "far"]
Wing = Literal["forehand", "backhand", "elbow"]
StrokeKind = Literal[
    "drive",
    "topspin",
    "block",
    "push",
    "chop",
    "serve",
]
Pace = Literal["controlled", "regular", "strong"]
StrokeDepth = Literal["short", "long"]

RPS = 2.0 * np.pi
TABLE_LEVEL = TABLE_HEIGHT + BALL_RADIUS
NET_LEVEL = TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
MAX_FLIGHT_HEIGHT_ABOVE_NET_MM = 50.0
MAX_REBOUND_HEIGHT_ABOVE_NET_MM = 25.0


@dataclass(frozen=True)
class ExerciseStroke:
    """One prescribed contact in an exercise."""

    hitter: Player
    wing: Wing
    kind: StrokeKind
    target_wing: Wing
    contact_wing: Wing | None = None
    pace: Pace = "regular"
    contact_moment: Literal[2, 3, 4] = 2
    contact_fraction: float = 0.45
    cycle: int = 0
    label: str = ""
    service_preset: str | None = None
    bounce_tolerance_mm: float = 150.0
    depth: StrokeDepth = "long"
    opening_attack: bool = False


@dataclass(frozen=True)
class ExerciseDefinition:
    """Declarative exercise expanded to a concrete list of contacts."""

    name: str
    title: str
    cycles: int
    strokes: tuple[ExerciseStroke, ...]


@dataclass(frozen=True)
class RallyValidation:
    passed: bool
    violations: tuple[str, ...]
    target_point: tuple[float, float]
    first_bounce: tuple[float, float] | None
    bounce_error_mm: float
    spin_error_rps: float
    net_clearance_mm: float
    outgoing_speed_mm_s: float
    depth_from_net_mm: float
    max_height_above_net_mm: float
    rebound_height_above_net_mm: float


@dataclass
class RallySegment:
    stroke: ExerciseStroke
    params: RacketImpactParameters
    result: SimulationResult
    stop_index: int
    validation: RallyValidation


@dataclass
class ExerciseResult:
    definition: ExerciseDefinition
    segments: tuple[RallySegment, ...]
    violations: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations and all(
            segment.validation.passed for segment in self.segments
        )

    @property
    def duration(self) -> float:
        return float(
            sum(
                segment.result.t[segment.stop_index]
                if segment.result.t is not None
                else segment.stop_index * 0.005
                for segment in self.segments
            )
        )


def opponent(player: Player) -> Player:
    return "far" if player == "near" else "near"


def travel_sign(hitter: Player) -> float:
    return 1.0 if hitter == "near" else -1.0


def side_name(player: Player) -> str:
    return "server" if player == "near" else "receiver"


def wing_y(player: Player, wing: Wing) -> float:
    """Resolve a right-handed player's wing to the global Y coordinate."""

    if wing == "elbow":
        return TABLE_WIDTH * 0.5
    near_value = TABLE_WIDTH * (0.28 if wing == "forehand" else 0.72)
    return near_value if player == "near" else TABLE_WIDTH - near_value


def stroke_spin_rps(stroke: ExerciseStroke) -> tuple[float, float, float]:
    """Return a global spin target whose sign follows travel direction."""

    sign = travel_sign(stroke.hitter)
    if stroke.kind == "drive":
        amount = {"controlled": 28.0, "regular": 36.0, "strong": 46.0}[stroke.pace]
        return (0.0, sign * amount, 0.0)
    if stroke.kind == "topspin":
        amount = {"controlled": 36.0, "regular": 46.0, "strong": 58.0}[stroke.pace]
        return (0.0, sign * amount, 0.0)
    if stroke.kind == "block":
        return (0.0, sign * 18.0, 0.0)
    if stroke.kind == "push":
        return (0.0, -sign * 35.0, 0.0)
    if stroke.kind == "chop":
        return (0.0, -sign * 48.0, 0.0)
    return (0.0, -sign * 45.0, 0.0)


def theoretical_contact(
    stroke: ExerciseStroke,
) -> tuple[Literal[2, 3, 4], float]:
    """Return the coaching phase required by the stroke's tactical role."""

    if stroke.kind == "drive":
        return 3, 0.5
    if stroke.kind == "topspin":
        return (4, 0.15) if stroke.opening_attack else (3, 0.5)
    if stroke.kind in {"block", "push"}:
        return 2, 0.45
    if stroke.kind == "chop":
        return 4, 0.55
    return stroke.contact_moment, stroke.contact_fraction


def stroke_target_point(stroke: ExerciseStroke) -> tuple[float, float]:
    """Return the first-bounce target for a non-service stroke."""

    receiver = opponent(stroke.hitter)
    sign = travel_sign(stroke.hitter)
    if stroke.depth == "short":
        depth_from_net = 290.0
    else:
        attack_depth = {
            "strong": 1120.0,
            "regular": 1020.0,
            "controlled": 900.0,
        }[stroke.pace]
        depth_from_net = {
            "drive": attack_depth,
            "topspin": attack_depth,
            "block": 980.0,
            "push": 1000.0,
            "chop": 1000.0,
        }.get(stroke.kind, 1000.0)
    x = TABLE_LENGTH / 2.0 + sign * depth_from_net
    return float(x), float(wing_y(receiver, stroke.target_wing))


def _arc_contact_index(
    result: SimulationResult,
    hitter: Player,
    moment: int,
    fraction: float,
) -> int:
    """Select a contact on the first post-bounce arc for either table side."""

    bounces = table_bounces(result, side_name(hitter))
    if not bounces:
        raise ValueError(f"No bounce is available on the {hitter} side.")
    first = bounces[0].index
    z = result.x[2]
    later_bounces = [event.index for event in bounces[1:]]
    if later_bounces:
        end = later_bounces[0]
    else:
        later = np.arange(first + 1, result.x.shape[1])
        arrivals = later[z[later] <= TABLE_LEVEL]
        end = int(arrivals[0]) if arrivals.size else result.x.shape[1] - 1
    arc = np.arange(first, end + 1)
    apex = int(arc[np.argmax(z[arc])])
    if moment == 3:
        return apex
    if moment == 2:
        candidates = np.arange(first + 1, apex)
    elif moment == 4:
        candidates = np.arange(apex + 1, end)
    else:
        raise ValueError(f"Unknown contact moment: {moment}")
    if not candidates.size:
        raise ValueError(
            f"Moment {moment} is unavailable on the {hitter} post-bounce arc."
        )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(candidates[round(fraction * (len(candidates) - 1))])


def _first_target_bounce(
    result: SimulationResult,
    target_player: Player,
):
    bounces = table_bounces(result, side_name(target_player))
    return bounces[0] if bounces else None


def _trajectory_metrics(
    position: tuple[float, float, float],
    velocity: np.ndarray,
    spin_rps: tuple[float, float, float],
    target_player: Player,
) -> tuple[np.ndarray, SimulationResult, object | None, float]:
    from .physics import simulate

    result = simulate(
        InitialConditions(
            pos=position,
            vel=tuple(float(value) for value in velocity),
            omega=tuple(float(value * RPS) for value in spin_rps),
        ),
        t_max=1.8,
    )
    bounce = _first_target_bounce(result, target_player)
    if bounce is None:
        all_bounces = table_bounces(result)
        bounce = all_bounces[0] if all_bounces else None
    crossings = net_crossings(result)
    clearance = (
        float(crossings[0].point[2] - NET_LEVEL)
        if crossings
        else float(
            result.x[
                2,
                int(np.argmin(np.abs(result.x[0] - TABLE_LENGTH / 2.0))),
            ]
            - NET_LEVEL
        )
    )
    point = (
        np.asarray(bounce.point[:2], dtype=float)
        if bounce is not None
        else np.array([np.nan, np.nan])
    )
    return point, result, bounce, clearance


def trajectory_height_metrics(
    result: SimulationResult,
    bounce,
    *,
    start_index: int = 0,
) -> tuple[float, float]:
    """Return the incoming-flight and first-rebound apex above the net."""

    if bounce is None:
        return float("inf"), float("inf")
    bounce_index = int(bounce.index)
    flight_start = max(0, min(int(start_index), bounce_index))
    flight_height = float(
        np.max(result.x[2, flight_start : bounce_index + 1]) - NET_LEVEL
    )
    later_bounces = [
        event.index
        for event in table_bounces(result)
        if event.index > bounce_index
    ]
    rebound_end = (
        int(later_bounces[0])
        if later_bounces
        else result.x.shape[1] - 1
    )
    rebound_height = float(
        np.max(result.x[2, bounce_index : rebound_end + 1]) - NET_LEVEL
    )
    return flight_height, rebound_height


def _launch_velocity(
    incoming: InitialConditions,
    stroke: ExerciseStroke,
    spin_override_rps: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, SimulationResult]:
    """Deterministically shoot a legal trajectory at the prescribed target."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - covered by environment doctor.
        raise RuntimeError(
            "Exercise calibration requires the search extra: "
            'pip install -e ".[search]"'
        ) from exc

    target = np.asarray(stroke_target_point(stroke), dtype=float)
    target_player = opponent(stroke.hitter)
    spin = spin_override_rps or stroke_spin_rps(stroke)
    sign = travel_sign(stroke.hitter)
    clearance_target = {
        "drive": 20.0,
        "topspin": 25.0,
        "block": 15.0,
        "push": 20.0,
        "chop": 30.0,
    }.get(stroke.kind, 20.0)
    distance = abs(target[0] - incoming.pos[0])
    horizontal_speed = {
        ("drive", "controlled"): 14000.0,
        ("drive", "regular"): 16000.0,
        ("drive", "strong"): 18500.0,
        ("topspin", "controlled"): 13800.0,
        ("topspin", "regular"): 16000.0,
        ("topspin", "strong"): 18500.0,
        ("block", "controlled"): 11500.0,
        ("block", "regular"): 13000.0,
        ("block", "strong"): 14500.0,
        ("push", "controlled"): 5000.0,
        ("push", "regular"): 5800.0,
        ("push", "strong"): 6500.0,
        ("chop", "controlled"): 5200.0,
        ("chop", "regular"): 6000.0,
        ("chop", "strong"): 6800.0,
    }.get((stroke.kind, stroke.pace), 8500.0)
    flight_time = float(np.clip(distance / horizontal_speed, 0.28, 0.72))
    vx = (target[0] - incoming.pos[0]) / flight_time
    vy = (target[1] - incoming.pos[1]) / flight_time
    vz = (
        TABLE_LEVEL
        - incoming.pos[2]
        + 0.5 * 9800.0 * flight_time**2
    ) / flight_time
    initial = np.array([vx, vy, max(250.0, vz)], dtype=float)
    lower = np.array(
        [1800.0, -9000.0, 100.0]
        if sign > 0
        else [-16000.0, -9000.0, 100.0]
    )
    upper = np.array(
        [16000.0, 9000.0, 8500.0]
        if sign > 0
        else [-1800.0, 9000.0, 8500.0]
    )
    initial = np.clip(initial, lower + 1.0, upper - 1.0)

    def residual(vector: np.ndarray) -> np.ndarray:
        point, trajectory, bounce, clearance = _trajectory_metrics(
            incoming.pos,
            vector,
            spin,
            target_player,
        )
        if bounce is None or not np.all(np.isfinite(point)):
            return np.array([25.0, 25.0, 25.0])
        flight_height, rebound_height = trajectory_height_metrics(
            trajectory,
            bounce,
        )
        return np.array(
            [
                (point[0] - target[0]) / 50.0,
                (point[1] - target[1]) / 50.0,
                (clearance - clearance_target) / 20.0,
                (np.linalg.norm(vector[:2]) - horizontal_speed) / 1800.0,
                max(0.0, flight_height - 45.0) / 5.0,
                max(0.0, rebound_height - 20.0) / 5.0,
            ]
        )

    fitted = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=90,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )
    velocity = np.asarray(fitted.x, dtype=float)
    _, result, bounce, _ = _trajectory_metrics(
        incoming.pos,
        velocity,
        spin,
        target_player,
    )
    if bounce is None:
        raise RuntimeError(f"Could not calibrate {stroke.label or stroke.kind}.")
    return velocity, result


def _racket_parameters(
    incoming: InitialConditions,
    desired_velocity: np.ndarray,
    stroke: ExerciseStroke,
) -> RacketImpactParameters:
    """Fit a Coulomb racket contact to a desired post-impact ball state."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Exercise calibration requires the search extra: "
            'pip install -e ".[search]"'
        ) from exc

    sign = travel_sign(stroke.hitter)
    target_spin = np.asarray(stroke_spin_rps(stroke), dtype=float) * RPS
    cut = stroke.kind in {"push", "chop"}
    friction = 0.82 if stroke.kind != "block" else 0.68
    restitution = 0.82
    angle_x = -35.0 if stroke.wing == "forehand" else 35.0
    best: tuple[float, RacketImpactParameters] | None = None

    def from_angles(angles: np.ndarray) -> RacketImpactParameters:
        racket_angle = (angle_x, float(angles[0]), float(angles[1]))
        normal = racket_normal(racket_angle)
        normal_projection = np.outer(normal, normal)
        tangent_projection = np.eye(3) - normal_projection
        impact_matrix = (
            (1.0 - friction) * tangent_projection
            - restitution * normal_projection
        )
        racket_matrix = np.eye(3) - impact_matrix
        racket_velocity = np.linalg.solve(
            racket_matrix,
            desired_velocity
            - impact_matrix @ np.asarray(incoming.vel, dtype=float),
        )
        return RacketImpactParameters(
            ball_velocity=incoming.vel,
            ball_omega=incoming.omega,
            rubber_friction=friction,
            rubber_restitution=restitution,
            racket_angle=racket_angle,
            racket_velocity=tuple(float(value) for value in racket_velocity),
            ball_position=incoming.pos,
            contact_model="legacy",
        )

    def residual(angles: np.ndarray) -> np.ndarray:
        params = from_angles(angles)
        post = apply_racket_impact(params)
        spin_error = (
            np.asarray(post.omega, dtype=float) - target_spin
        ) / (RPS * 10.0)
        racket_velocity = np.asarray(params.racket_velocity)
        vertical_penalty = (
            max(0.0, (racket_velocity[2] + 120.0) / 25.0)
            if cut
            else max(0.0, (120.0 - racket_velocity[2]) / 100.0)
        )
        forward_penalty = max(
            0.0,
            (150.0 - sign * racket_velocity[0]) / 1000.0,
        )
        return np.concatenate(
            [spin_error, [vertical_penalty, forward_penalty]]
        )

    z_guesses = (0.0, 35.0, -35.0) if sign > 0 else (180.0, 145.0, -145.0)
    y_guesses = (-45.0, 0.0, 45.0)
    for angle_y in y_guesses:
        for angle_z in z_guesses:
            guess = np.array([angle_y, angle_z])
            fitted = least_squares(
                residual,
                guess,
                bounds=((-85.0, -180.0), (85.0, 180.0)),
                max_nfev=120,
            )
            params = from_angles(fitted.x)
            score = float(np.linalg.norm(residual(fitted.x)))
            if best is None or score < best[0]:
                best = score, params
    assert best is not None
    # The ACE table transition can feed substantially more residual spin into
    # the next contact. Keep the inversion bounded, then let the full segment
    # validator enforce the actual 35 rps tolerance and gesture constraints.
    if best[0] > 50.0:
        raise RuntimeError(
            f"Racket inversion failed for {stroke.label or stroke.kind}: "
            f"scaled error={best[0]:.3f}"
        )
    return best[1]


def _jointly_calibrated_racket_parameters(
    incoming: InitialConditions,
    stroke: ExerciseStroke,
) -> RacketImpactParameters:
    """Fit trajectory and legacy racket contact in one coupled solve."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Exercise calibration requires the search extra: "
            'pip install -e ".[search]"'
        ) from exc

    target = np.asarray(stroke_target_point(stroke), dtype=float)
    target_player = opponent(stroke.hitter)
    target_spin = np.asarray(stroke_spin_rps(stroke), dtype=float)
    sign = travel_sign(stroke.hitter)
    cut = stroke.kind in {"push", "chop"}
    friction = 0.68 if stroke.kind == "block" else 0.82
    restitution = 0.82
    angle_x = -35.0 if stroke.wing == "forehand" else 35.0
    desired_velocity, _ = _launch_velocity(incoming, stroke)
    initial_params = _racket_parameters(incoming, desired_velocity, stroke)
    initial = np.array(
        [
            *desired_velocity,
            initial_params.racket_angle[1],
            initial_params.racket_angle[2],
        ],
        dtype=float,
    )
    lower = np.array(
        [1800.0, -12000.0, 100.0, -85.0, -180.0]
        if sign > 0
        else [-24000.0, -12000.0, 100.0, -85.0, -180.0]
    )
    upper = np.array(
        [24000.0, 12000.0, 8500.0, 85.0, 180.0]
        if sign > 0
        else [-1800.0, 12000.0, 8500.0, 85.0, 180.0]
    )
    initial = np.clip(initial, lower + 1e-5, upper - 1e-5)

    def parameters(values: np.ndarray) -> RacketImpactParameters:
        desired = np.asarray(values[:3], dtype=float)
        racket_angle = (
            angle_x,
            float(values[3]),
            float(values[4]),
        )
        normal = racket_normal(racket_angle)
        normal_projection = np.outer(normal, normal)
        tangent_projection = np.eye(3) - normal_projection
        impact_matrix = (
            (1.0 - friction) * tangent_projection
            - restitution * normal_projection
        )
        racket_matrix = np.eye(3) - impact_matrix
        racket_velocity = np.linalg.solve(
            racket_matrix,
            desired
            - impact_matrix @ np.asarray(incoming.vel, dtype=float),
        )
        return RacketImpactParameters(
            ball_velocity=incoming.vel,
            ball_omega=incoming.omega,
            rubber_friction=friction,
            rubber_restitution=restitution,
            racket_angle=racket_angle,
            racket_velocity=tuple(float(value) for value in racket_velocity),
            ball_position=incoming.pos,
            contact_model="legacy",
        )

    def residual(values: np.ndarray) -> np.ndarray:
        params = parameters(values)
        post = apply_racket_impact(params)
        result = simulate_racket_impact(params, t_max=1.8)
        target_bounce = _first_target_bounce(result, target_player)
        bounce = target_bounce
        if bounce is None:
            all_bounces = table_bounces(result)
            bounce = all_bounces[0] if all_bounces else None
        if bounce is None:
            return np.full(12, 50.0)
        point = np.asarray(bounce.point[:2], dtype=float)
        crossings = net_crossings(result)
        clearance = (
            float(crossings[0].point[2] - NET_LEVEL)
            if crossings
            else -200.0
        )
        flight_height, rebound_height = trajectory_height_metrics(
            result,
            bounce,
        )
        actual_spin = np.asarray(post.omega, dtype=float) / RPS
        racket_velocity = np.asarray(params.racket_velocity, dtype=float)
        vertical_penalty = (
            max(0.0, (racket_velocity[2] + 1.0) / 25.0)
            if cut
            else max(0.0, (51.0 - racket_velocity[2]) / 25.0)
        )
        forward_penalty = max(
            0.0,
            (151.0 - sign * racket_velocity[0]) / 250.0,
        )
        depth = abs(float(point[0]) - TABLE_LENGTH / 2.0)
        depth_penalty = (
            max(0.0, depth - 450.0) / 25.0
            if stroke.depth == "short"
            else max(0.0, 800.0 - depth) / 25.0
        )
        return np.concatenate(
            [
                (point - target) / 60.0,
                [(clearance - 25.0) / 1.5],
                (actual_spin - target_spin) / 20.0,
                [
                    max(0.0, flight_height - 35.0) / 1.5,
                    max(0.0, rebound_height - 15.0) / 1.5,
                    depth_penalty,
                    vertical_penalty,
                    forward_penalty,
                    0.0 if target_bounce is not None else 20.0,
                ],
            ]
        )

    fitted = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=240,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
        x_scale="jac",
    )
    best_values = np.asarray(fitted.x, dtype=float)
    best_params = parameters(best_values)
    best_result = simulate_racket_impact(best_params, t_max=1.8)
    best_validation = validate_rally_segment(
        stroke,
        best_params,
        best_result,
    )
    if best_validation.passed:
        return best_params

    from scipy.optimize import differential_evolution

    seed_material = (
        f"{stroke.hitter}/{stroke.wing}/{stroke.kind}/{stroke.target_wing}/"
        f"{stroke.pace}/{incoming.pos[0]:.1f}/{incoming.pos[1]:.1f}"
    )

    def objective(values: np.ndarray) -> float:
        errors = residual(values)
        return float(np.dot(errors, errors))

    global_fit = differential_evolution(
        objective,
        bounds=tuple(zip(lower, upper)),
        seed=zlib.crc32(seed_material.encode("utf-8")) & 0xFFFFFFFF,
        popsize=6,
        maxiter=50,
        tol=0.005,
        polish=True,
        x0=best_values,
        workers=1,
        updating="immediate",
    )
    global_params = parameters(np.asarray(global_fit.x, dtype=float))
    global_result = simulate_racket_impact(global_params, t_max=1.8)
    global_validation = validate_rally_segment(
        stroke,
        global_params,
        global_result,
    )
    if global_validation.passed:
        return global_params
    if _rally_validation_score(global_validation) < _rally_validation_score(
        best_validation
    ):
        return global_params
    return best_params


def validate_rally_segment(
    stroke: ExerciseStroke,
    params: RacketImpactParameters,
    result: SimulationResult,
    spin_tolerance_rps: float | None = None,
) -> RallyValidation:
    """Validate one directional rally stroke before the next contact."""

    violations: list[str] = []
    receiver = opponent(stroke.hitter)
    target = stroke_target_point(stroke)
    bounce = _first_target_bounce(result, receiver)
    first_bounce = None if bounce is None else tuple(
        float(value) for value in bounce.point[:2]
    )
    error = (
        float(np.linalg.norm(np.asarray(first_bounce) - np.asarray(target)))
        if first_bounce is not None
        else float("inf")
    )
    if bounce is None:
        violations.append("shot has no bounce on the opponent half")
    if error > stroke.bounce_tolerance_mm:
        violations.append("shot bounce is outside tolerance")
    depth_from_net = (
        abs(first_bounce[0] - TABLE_LENGTH / 2.0)
        if first_bounce is not None
        else float("-inf")
    )
    if stroke.depth == "short" and depth_from_net > 450.0:
        violations.append("short shot does not land near the net")
    if stroke.depth == "long" and depth_from_net < 800.0:
        violations.append("long shot does not reach the back of the table")
    flight_height, rebound_height = trajectory_height_metrics(result, bounce)
    if flight_height > MAX_FLIGHT_HEIGHT_ABOVE_NET_MM:
        violations.append("shot apex is more than 50 mm above the net")
    if rebound_height > MAX_REBOUND_HEIGHT_ABOVE_NET_MM:
        violations.append("first rebound apex is more than 25 mm above the net")
    crossings = net_crossings(result)
    clearance = (
        float(crossings[0].point[2] - NET_LEVEL)
        if crossings
        else float("-inf")
    )
    if not crossings:
        violations.append("shot does not cross the net")
    cutoff = bounce.time if bounce is not None else float("inf")
    if any(
        event.kind == "net_contact" and event.time <= cutoff
        for event in result.events
    ):
        violations.append("shot touches the net")
    if clearance < 5.0:
        violations.append("shot net clearance is below 5 mm")
    post = apply_racket_impact(params)
    desired_spin = np.asarray(stroke_spin_rps(stroke))
    actual_spin = np.asarray(post.omega) / RPS
    spin_error = float(np.linalg.norm(actual_spin - desired_spin))
    resolved_spin_tolerance = (
        stroke_spin_tolerance_rps(stroke)
        if spin_tolerance_rps is None
        else spin_tolerance_rps
    )
    if spin_error > resolved_spin_tolerance:
        violations.append("shot spin is outside tolerance")
    if travel_sign(stroke.hitter) * post.vel[0] <= 0:
        violations.append("shot travels in the wrong direction")
    if stroke.kind in {"drive", "topspin"} and params.racket_velocity[2] <= 25.0:
        violations.append("attacking racket does not move upward")
    if stroke.kind in {"push", "chop"} and params.racket_velocity[2] >= 0.0:
        violations.append("cutting racket does not move downward")
    return RallyValidation(
        passed=not violations,
        violations=tuple(violations),
        target_point=target,
        first_bounce=first_bounce,
        bounce_error_mm=error,
        spin_error_rps=spin_error,
        net_clearance_mm=clearance,
        outgoing_speed_mm_s=float(np.linalg.norm(post.vel)),
        depth_from_net_mm=float(depth_from_net),
        max_height_above_net_mm=flight_height,
        rebound_height_above_net_mm=rebound_height,
    )


def stroke_spin_tolerance_rps(stroke: ExerciseStroke) -> float:
    """Return the calibrated spin tolerance for an exercise stroke."""

    return {
        "drive": 210.0,
        "topspin": 170.0,
        "block": 100.0,
        "push": 45.0,
        "chop": 90.0,
        "serve": 45.0,
    }[stroke.kind]


def _rally_validation_score(validation: RallyValidation) -> float:
    bounce_error = (
        validation.bounce_error_mm
        if np.isfinite(validation.bounce_error_mm)
        else 10000.0
    )
    clearance_penalty = (
        max(0.0, 5.0 - validation.net_clearance_mm) * 100.0
        if np.isfinite(validation.net_clearance_mm)
        else 10000.0
    )
    return (
        10000.0 * len(validation.violations)
        + bounce_error
        + 5.0 * validation.spin_error_rps
        + clearance_penalty
    )


def _initial_incoming(stroke: ExerciseStroke) -> InitialConditions:
    x = -180.0 if stroke.hitter == "near" else TABLE_LENGTH + 180.0
    direction = -travel_sign(stroke.hitter)
    moment, fraction = theoretical_contact(stroke)
    if moment == 2:
        vertical_speed = 650.0
        contact_height = TABLE_HEIGHT + 140.0
    elif moment == 3:
        vertical_speed = 0.0
        contact_height = TABLE_HEIGHT + 190.0
    else:
        vertical_speed = -max(180.0, 650.0 * fraction)
        contact_height = TABLE_HEIGHT + 160.0
    return InitialConditions(
        pos=(
            x,
            wing_y(stroke.hitter, stroke.contact_wing or stroke.wing),
            contact_height,
        ),
        vel=(direction * 2600.0, 0.0, vertical_speed),
        omega=(0.0, direction * 18.0 * RPS, 0.0),
    )


def _service_parameters(name: str) -> RacketImpactParameters:
    if name == "short":
        from .presets.exercises import SHORT_EXERCISE_SERVICE_PARAMS

        return SHORT_EXERCISE_SERVICE_PARAMS
    if name in {"long", "long_backhand"}:
        from .presets.exercises import LONG_EXERCISE_SERVICE_PARAMS

        return LONG_EXERCISE_SERVICE_PARAMS
    if name == "long_forehand":
        from .presets.exercises import (
            LONG_FOREHAND_EXERCISE_SERVICE_PARAMS,
        )

        return LONG_FOREHAND_EXERCISE_SERVICE_PARAMS
    raise ValueError(f"Unknown exercise service preset: {name}")


def simulate_exercise(
    definition: ExerciseDefinition,
    *,
    dt: float = 0.005,
    t_max: float = 1.8,
    use_calibrated_preset: bool = True,
) -> ExerciseResult:
    """Calibrate, simulate, validate, and chain every stroke in an exercise."""

    segments: list[RallySegment] = []
    incoming: InitialConditions | None = None
    violations: list[str] = []
    for stroke_index, stroke in enumerate(definition.strokes, start=1):
        if stroke.kind == "serve":
            continue
        expected_moment, expected_fraction = theoretical_contact(stroke)
        if (
            stroke.contact_moment != expected_moment
            or not np.isclose(
                stroke.contact_fraction,
                expected_fraction,
                atol=1e-9,
            )
        ):
            violations.append(
                f"stroke {stroke_index}: contact selection violates "
                "the theoretical timing rule"
            )
    preset_controls = None
    if use_calibrated_preset and definition.cycles == 3:
        from .presets.exercises import CALIBRATED_CONTROLS

        candidate = CALIBRATED_CONTROLS.get(definition.name)
        if candidate is not None and len(candidate) == len(definition.strokes):
            preset_controls = candidate
    for index, stroke in enumerate(definition.strokes):
        if stroke.kind == "serve":
            params = _service_parameters(stroke.service_preset or "short")
            result = simulate_racket_impact(params, dt=dt, t_max=t_max)
            from .exchange import ServiceTargets
            from .validation import validate_service

            service_targets = (
                ServiceTargets()
                if (stroke.service_preset or "short") == "short"
                else ServiceTargets(
                    depth="long",
                    direction=(
                        "forehand"
                        if stroke.service_preset == "long_forehand"
                        else "backhand"
                    ),
                    spin_rps=(0.0, -45.0, 35.0),
                    bounce_tolerance_mm=100.0,
                    spin_tolerance_rps=12.0,
                    max_height_above_net_mm=50.0,
                    max_rebound_height_above_net_mm=25.0,
                )
            )
            service_report = validate_service(
                params,
                result,
                service_targets,
            )
            receiver_bounces = table_bounces(result, "receiver")
            service_bounce = receiver_bounces[0] if receiver_bounces else None
            server_bounces = table_bounces(result, "server")
            service_flight_start = (
                server_bounces[0].index if server_bounces else 0
            )
            flight_height, rebound_height = trajectory_height_metrics(
                result,
                service_bounce,
                start_index=service_flight_start,
            )
            service_violations = list(service_report.violations)
            if flight_height > MAX_FLIGHT_HEIGHT_ABOVE_NET_MM:
                service_violations.append(
                    "service apex is more than 50 mm above the net"
                )
            if rebound_height > MAX_REBOUND_HEIGHT_ABOVE_NET_MM:
                service_violations.append(
                    "service rebound apex is more than 25 mm above the net"
                )
            validation = RallyValidation(
                passed=not service_violations,
                violations=tuple(service_violations),
                target_point=service_report.target_point,
                first_bounce=service_report.first_bounce,
                bounce_error_mm=service_report.bounce_error_mm,
                spin_error_rps=service_report.spin_error_rps,
                net_clearance_mm=service_report.net_clearance_mm,
                outgoing_speed_mm_s=float(
                    np.linalg.norm(apply_racket_impact(params).vel)
                ),
                depth_from_net_mm=(
                    abs(
                        service_report.first_bounce[0]
                        - TABLE_LENGTH / 2.0
                    )
                    if service_report.first_bounce is not None
                    else float("-inf")
                ),
                max_height_above_net_mm=flight_height,
                rebound_height_above_net_mm=rebound_height,
            )
        else:
            if incoming is None:
                incoming = _initial_incoming(stroke)
            if preset_controls is not None:
                angle_y, angle_z, racket_x, racket_y, racket_z = (
                    preset_controls[index]
                )
                params = RacketImpactParameters(
                    ball_velocity=incoming.vel,
                    ball_omega=incoming.omega,
                    rubber_friction=(
                        0.68 if stroke.kind == "block" else 0.82
                    ),
                    rubber_restitution=0.82,
                    racket_angle=(
                        -35.0 if stroke.wing == "forehand" else 35.0,
                        angle_y,
                        angle_z,
                    ),
                    racket_velocity=(racket_x, racket_y, racket_z),
                    ball_position=incoming.pos,
                    contact_model="legacy",
                )
            else:
                params = _jointly_calibrated_racket_parameters(
                    incoming,
                    stroke,
                )
            result = simulate_racket_impact(params, dt=dt, t_max=t_max)
            validation = validate_rally_segment(stroke, params, result)
        if index + 1 < len(definition.strokes):
            next_stroke = definition.strokes[index + 1]
            try:
                stop_index = _arc_contact_index(
                    result,
                    next_stroke.hitter,
                    next_stroke.contact_moment,
                    next_stroke.contact_fraction,
                )
            except ValueError as exc:
                stop_index = result.x.shape[1] - 1
                violations.append(f"stroke {index + 1}: {exc}")
                segments.append(
                    RallySegment(stroke, params, result, stop_index, validation)
                )
                break
            incoming = InitialConditions(
                pos=tuple(float(value) for value in result.x[:, stop_index]),
                vel=tuple(float(value) for value in result.v[:, stop_index]),
                omega=tuple(float(value) for value in result.omega[:, stop_index]),
            )
        else:
            below_floor = np.where(result.x[2] < 0.0)[0]
            stop_index = (
                int(below_floor[0])
                if below_floor.size
                else result.x.shape[1] - 1
            )
        segments.append(
            RallySegment(stroke, params, result, stop_index, validation)
        )
        if not validation.passed:
            violations.extend(
                f"stroke {index + 1}: {message}"
                for message in validation.violations
            )

    for index in range(len(segments) - 1):
        current = segments[index]
        following = segments[index + 1]
        expected = current.result.x[:, current.stop_index]
        actual = np.asarray(following.params.ball_position)
        if not np.allclose(expected, actual, atol=1e-6):
            violations.append(
                f"continuity failure between strokes {index + 1} and {index + 2}"
            )
        if following.params.ball_velocity != tuple(
            float(value) for value in current.result.v[:, current.stop_index]
        ):
            violations.append(
                f"velocity discontinuity before stroke {index + 2}"
            )
        if following.params.ball_omega != tuple(
            float(value) for value in current.result.omega[:, current.stop_index]
        ):
            violations.append(
                f"spin discontinuity before stroke {index + 2}"
            )
    return ExerciseResult(
        definition=definition,
        segments=tuple(segments),
        violations=tuple(violations),
    )


def with_cycles(
    definition: ExerciseDefinition,
    cycles: int,
) -> ExerciseDefinition:
    """Return a definition rebuilt by its preset factory."""

    from .presets.exercises import build_exercise

    return build_exercise(definition.name, cycles=cycles)


__all__ = [
    "ExerciseDefinition",
    "ExerciseResult",
    "ExerciseStroke",
    "RallySegment",
    "RallyValidation",
    "simulate_exercise",
    "stroke_spin_tolerance_rps",
    "stroke_spin_rps",
    "stroke_target_point",
    "theoretical_contact",
    "travel_sign",
    "wing_y",
    "with_cycles",
]
