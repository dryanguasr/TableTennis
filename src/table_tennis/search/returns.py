"""Search and validate a complete serve-return exchange.

The public API in this module keeps the incoming service immutable while a
return is searched.  Rubber properties are fixed inputs; the optimizer changes
contact timing, racket orientation and racket velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np

from ..constants import (
    BALL_MASS,
    BALL_RADIUS,
    BALL_ROT_INERTIA,
    G,
    NET_HEIGHT,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from ..events import table_bounces
from ..exchange import (
    ContactSelection,
    ExchangeResult,
    RubberProperties,
    ServiceTargets,
    StrokeTargets,
    ValidationReport,
    contact_index,
    contact_state,
    simulate_exchange,
)
from ..models import (
    InitialConditions,
    RacketImpactParameters,
    SimulationResult,
)
from ..physics import (
    apply_racket_impact,
    simulate_racket_impact,
)
from ..presets.returns import PILOT_SERVICE_PARAMS, RETURN_PRESET_VECTORS
from ..validation import validate_return, validate_service

try:
    from scipy.optimize import differential_evolution, minimize
except Exception:  # pragma: no cover - depends on the user's environment.
    differential_evolution = None
    minimize = None


RPS = 2 * np.pi
NET_CLEARANCE_LEVEL = TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
@dataclass(frozen=True)
class ReturnSearchConfig:
    maxiter: int = 180
    popsize: int = 18
    restarts: int = 5
    seed: int = 7
    workers: int = 1
    polish: bool = True
    dt: float = 0.005
    t_max: float = 3.0


@dataclass
class ReturnSearchResult:
    success: bool
    cost: float
    vector: tuple[float, ...]
    params: RacketImpactParameters
    trajectory: SimulationResult
    validation: ValidationReport
    contact: ContactSelection | None = None
    contact_index: int | None = None
    optimizer: str = ""
    message: str = ""


def _unused_table_bounces(result: SimulationResult, side: str | None = None):
    events = [event for event in result.events if event.kind == "table_bounce"]
    return [event for event in events if side is None or event.side == side]


def _unused_net_crossings(result: SimulationResult):
    return [event for event in result.events if event.kind == "net_cross"]


def _first_bounce_xy(result: SimulationResult) -> tuple[float, float] | None:
    bounces = table_bounces(result)
    if not bounces:
        return None
    return float(bounces[0].point[0]), float(bounces[0].point[1])


def _net_clearance(result: SimulationResult) -> float:
    crossings = net_crossings(result)
    if not crossings:
        return float("-inf")
    return float(crossings[0].point[2] - NET_CLEARANCE_LEVEL)


def _spin_error(params: RacketImpactParameters, target_spin_rps: Iterable[float]) -> float:
    actual = np.asarray(apply_racket_impact(params).omega, dtype=float) / RPS
    return float(np.linalg.norm(actual - np.asarray(tuple(target_spin_rps), dtype=float)))


def _depth_bounce_violation(depth: str, count: int) -> str | None:
    if depth in {"short", "two_bounce"} and count < 2:
        return "the shot does not bounce twice on the target half"
    if depth == "long" and count != 1:
        return "the long shot does not leave the table after its first bounce"
    return None


def _unused_validate_service(
    params: RacketImpactParameters,
    result: SimulationResult,
    targets: ServiceTargets = ServiceTargets(),
) -> ValidationReport:
    violations = []
    bounces = table_bounces(result)
    receiver_bounces = [
        event
        for event in table_bounces(result, "receiver")
        if event.point[0] < TABLE_LENGTH - BALL_RADIUS
    ]
    first_bounce = _first_bounce_xy(result)
    target = targets.target_point
    bounce_error = (
        float(np.linalg.norm(np.asarray(first_bounce) - np.asarray(target)))
        if first_bounce is not None and bounces and bounces[0].side == "receiver"
        else float("inf")
    )
    if len(bounces) < 2 or bounces[0].side != "server" or bounces[1].side != "receiver":
        violations.append("service bounces are not server then receiver")
    else:
        receiver_point = np.asarray(bounces[1].point[:2], dtype=float)
        bounce_error = float(np.linalg.norm(receiver_point - np.asarray(target)))
        first_bounce = tuple(float(value) for value in receiver_point)
    if bounce_error > targets.bounce_tolerance_mm:
        violations.append("service bounce is outside tolerance")
    if params.ball_position[0] >= 0 or params.ball_position[2] <= TABLE_HEIGHT:
        violations.append("service contact is not behind the baseline and above the table")
    if abs(params.ball_velocity[0]) > 1e-6 or abs(params.ball_velocity[1]) > 1e-6:
        violations.append("service toss is not vertical")
    minimum_toss_speed = np.sqrt(2 * G * 160.0)
    maximum_toss_speed = np.sqrt(2 * G * 2000.0)
    if not minimum_toss_speed <= abs(params.ball_velocity[2]) <= maximum_toss_speed:
        violations.append("service toss is outside the 16 cm to 2 m range")
    if np.linalg.norm(params.ball_omega) > 1e-6:
        violations.append("service toss imparts spin before racket contact")
    service_cutoff = bounces[1].time if len(bounces) >= 2 else float("inf")
    if any(event.kind == "net_contact" and event.time <= service_cutoff for event in result.events):
        violations.append("service touches the net")
    clearance = _net_clearance(result)
    if clearance < targets.min_net_clearance_mm:
        violations.append("service net clearance is too small")
    spin_error = _spin_error(params, targets.spin_rps)
    if spin_error > targets.spin_tolerance_rps:
        violations.append("service spin is outside tolerance")
    depth_violation = _depth_bounce_violation(targets.depth, len(receiver_bounces))
    if depth_violation:
        violations.append(depth_violation)
    return ValidationReport(
        passed=not violations,
        violations=tuple(violations),
        target_point=target,
        first_bounce=first_bounce,
        bounce_error_mm=bounce_error,
        spin_error_rps=spin_error,
        net_clearance_mm=clearance,
        bounces_on_target_side=len(receiver_bounces),
    )


def _unused_validate_return(
    params: RacketImpactParameters,
    result: SimulationResult,
    targets: StrokeTargets,
) -> ValidationReport:
    violations = []
    bounces = table_bounces(result)
    server_bounces = [
        event
        for event in table_bounces(result, "server")
        if event.point[0] > BALL_RADIUS
    ]
    if targets.depth == "long" and server_bounces:
        first_index = server_bounces[0].index
        later = np.arange(first_index + 1, result.x.shape[1])
        exits = later[
            (result.x[0, later] <= BALL_RADIUS)
            | (result.x[1, later] <= 0.0)
            | (result.x[1, later] >= TABLE_WIDTH)
            | (result.x[2, later] < 0.0)
        ]
        if exits.size:
            server_bounces = [
                event for event in server_bounces if event.index < int(exits[0])
            ]
    first_bounce = _first_bounce_xy(result)
    target = targets.target_point
    bounce_error = (
        float(np.linalg.norm(np.asarray(first_bounce) - np.asarray(target)))
        if first_bounce is not None
        else float("inf")
    )
    if not bounces or bounces[0].side != "server":
        violations.append("return does not first bounce on the server half")
    if bounce_error > targets.bounce_tolerance_mm:
        violations.append("return bounce is outside tolerance")
    if apply_racket_impact(params).vel[0] >= 0:
        violations.append("return does not travel toward the server")
    return_cutoff = bounces[0].time if bounces else float("inf")
    if any(event.kind == "net_contact" and event.time <= return_cutoff for event in result.events):
        violations.append("return touches the net")
    clearance = _net_clearance(result)
    if clearance < targets.min_net_clearance_mm:
        violations.append("return net clearance is too small")
    cut_stroke = targets.spin_rps[1] > 0
    maximum_clearance = (
        targets.max_net_clearance_mm
        if targets.max_net_clearance_mm is not None
        else (180.0 if cut_stroke else 350.0)
    )
    if clearance > maximum_clearance:
        violations.append("return net clearance is unrealistically high")
    first_bounce_index = bounces[0].index if bounces else result.x.shape[1] - 1
    maximum_height = float(
        np.max(result.x[2, : first_bounce_index + 1]) - (TABLE_HEIGHT + BALL_RADIUS)
    )
    allowed_height = (
        targets.max_height_above_table_mm
        if targets.max_height_above_table_mm is not None
        else (420.0 if cut_stroke else 650.0)
    )
    if maximum_height > allowed_height:
        violations.append("return arc is unrealistically high")
    post_impact = apply_racket_impact(params)
    if cut_stroke:
        if params.racket_velocity[0] >= 0 or params.racket_velocity[2] >= -100.0:
            violations.append("cut stroke racket must move forward and downward")
        if params.ball_position[2] < NET_CLEARANCE_LEVEL and post_impact.vel[2] <= 0:
            violations.append("cut return cannot rise from contact to net height")
        if post_impact.vel[2] > 1500.0:
            violations.append("cut return launches the ball too steeply upward")
    elif params.racket_velocity[2] <= 100.0:
        violations.append("topspin stroke racket must move upward")
    if bounces and result.v[0, bounces[0].index] >= 0:
        violations.append("return reverses direction after its first bounce")
    if len(server_bounces) >= 2 and server_bounces[1].point[0] >= server_bounces[0].point[0]:
        violations.append("return bounces back toward the net")
    spin_error = _spin_error(params, targets.spin_rps)
    if spin_error > targets.spin_tolerance_rps:
        violations.append("return spin is outside tolerance")
    depth_violation = _depth_bounce_violation(targets.depth, len(server_bounces))
    if depth_violation:
        violations.append(depth_violation)
    reported_bounces = min(len(server_bounces), 2) if targets.depth in {"short", "two_bounce"} else len(server_bounces)
    return ValidationReport(
        passed=not violations,
        violations=tuple(violations),
        target_point=target,
        first_bounce=first_bounce,
        bounce_error_mm=bounce_error,
        spin_error_rps=spin_error,
        net_clearance_mm=clearance,
        bounces_on_target_side=reported_bounces,
    )


def _unused_contact_index(
    service_result: SimulationResult,
    selection: ContactSelection,
) -> int:
    moments = identify_trajectory_moments(service_result)
    if selection.moment not in moments:
        raise ValueError(f"Moment {selection.moment} is not available for this service.")
    moment = moments[selection.moment]
    if selection.moment == 3 or moment.interval is None:
        return moment.index
    fraction = float(np.clip(selection.fraction, 0.0, 1.0))
    target_time = moment.interval[0] + fraction * (moment.interval[1] - moment.interval[0])
    times = service_result.t
    if times is None:
        return moment.index
    return int(np.argmin(np.abs(times - target_time)))


def _unused_contact_state(
    service_result: SimulationResult,
    selection: ContactSelection,
) -> tuple[int, InitialConditions]:
    index = _unused_contact_index(service_result, selection)
    return index, InitialConditions(
        pos=tuple(float(value) for value in service_result.x[:, index]),
        vel=tuple(float(value) for value in service_result.v[:, index]),
        omega=tuple(float(value) for value in service_result.omega[:, index]),
    )


def _normal_basis(delta_spin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = delta_spin / np.linalg.norm(delta_spin)
    reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


def racket_parameters_for_target_spin(
    incoming: InitialConditions,
    target_spin_rps: Iterable[float],
    normal_phase: float,
    normal_impulse: float,
    rubber: RubberProperties,
    stroke_side: str,
) -> RacketImpactParameters:
    """Invert the impact equations while imposing the requested post-impact spin."""

    ball_v = np.asarray(incoming.vel, dtype=float)
    ball_w = np.asarray(incoming.omega, dtype=float)
    target_w = np.asarray(tuple(target_spin_rps), dtype=float) * RPS
    delta_spin = target_w - ball_w
    if np.linalg.norm(delta_spin) < 1e-8:
        delta_spin = np.array([0.0, 0.0, 1e-8])
    first, second = _normal_basis(delta_spin)
    normal = np.cos(normal_phase) * first + np.sin(normal_phase) * second

    friction = float(rubber.friction)
    restitution = float(rubber.restitution)
    if friction <= 0:
        raise ValueError("Rubber friction must be greater than zero for impact inversion.")
    delta_spin = target_w - ball_w
    tangential_delta_v = (
        BALL_ROT_INERTIA / (BALL_MASS * BALL_RADIUS)
    ) * np.cross(normal, delta_spin)
    normal_delta_v = normal_impulse * normal
    post_velocity = ball_v + normal_delta_v + tangential_delta_v

    # Choose the racket velocity that produces the required sticking impulse.
    # The optimizer must still supply enough normal impulse for the Coulomb cap;
    # validate_return catches candidates that cannot reach the requested spin.
    effective_tangent_factor = 1.0 + BALL_MASS * BALL_RADIUS**2 / BALL_ROT_INERTIA
    required_slip = -effective_tangent_factor * tangential_delta_v
    contact_radius = -BALL_RADIUS * normal
    relative_tangent = required_slip - np.cross(ball_w, contact_radius)
    ball_normal_speed = float(np.dot(ball_v, normal))
    relative_normal_speed = -normal_impulse / (1.0 + restitution)
    racket_tangent = ball_v - ball_normal_speed * normal - relative_tangent
    racket_velocity = racket_tangent + (
        ball_normal_speed - relative_normal_speed
    ) * normal

    angle_y = -np.degrees(np.arcsin(np.clip(normal[2], -1.0, 1.0)))
    angle_z = np.degrees(np.arctan2(normal[1], normal[0]))
    angle_x = -35.0 if stroke_side == "forehand" else 35.0
    return RacketImpactParameters(
        ball_velocity=tuple(ball_v),
        ball_omega=tuple(ball_w),
        rubber_friction=friction,
        rubber_restitution=restitution,
        racket_angle=(angle_x, float(angle_y), float(angle_z)),
        racket_velocity=tuple(float(value) for value in racket_velocity),
        ball_position=incoming.pos,
    )


def return_params_from_vector(
    service_result: SimulationResult,
    targets: StrokeTargets,
    vector: Iterable[float],
    rubber: RubberProperties = RubberProperties(),
) -> tuple[ContactSelection, int, RacketImpactParameters]:
    values = np.asarray(tuple(vector), dtype=float)
    if len(values) != 6:
        raise ValueError("Return vectors contain fraction, two racket angles and three velocity components.")
    selection = ContactSelection(moment=4, fraction=float(values[0]))
    index, incoming = contact_state(service_result, selection)
    params = RacketImpactParameters(
        ball_velocity=incoming.vel,
        ball_omega=incoming.omega,
        rubber_friction=rubber.friction,
        rubber_restitution=rubber.restitution,
        racket_angle=(
            -35.0 if targets.stroke_side == "forehand" else 35.0,
            float(values[1]),
            float(values[2]),
        ),
        racket_velocity=tuple(float(value) for value in values[3:6]),
        ball_position=incoming.pos,
    )
    return selection, index, params


def build_return_preset(
    name: str,
    stroke_side: Literal["forehand", "backhand"],
    service_result: SimulationResult | None = None,
) -> tuple[ContactSelection, int, RacketImpactParameters]:
    if name not in RETURN_PRESET_VECTORS:
        raise ValueError(f"Unknown return preset: {name}")
    if service_result is None:
        service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=2.0)
    depth = {
        "cut_short": "short",
        "cut_two_bounce": "two_bounce",
        "cut_long": "long",
        "top_two_bounce": "two_bounce",
        "top_long": "long",
    }[name]
    spin = (-15.0, 35.0, 10.0) if name.startswith("cut_") else (0.0, -45.0, 0.0)
    targets = StrokeTargets(depth=depth, spin_rps=spin, stroke_side=stroke_side)
    return return_params_from_vector(service_result, targets, RETURN_PRESET_VECTORS[name])


def _unused_simulate_exchange(
    service_params: RacketImpactParameters,
    return_params: RacketImpactParameters,
    contact: ContactSelection,
    service_targets: ServiceTargets,
    return_targets: StrokeTargets,
    dt: float = 0.005,
    t_max: float = 3.0,
) -> ExchangeResult:
    service_result = simulate_racket_impact(service_params, dt=dt, t_max=t_max)
    index = contact_index(service_result, contact)
    expected_position = service_result.x[:, index]
    if not np.allclose(expected_position, return_params.ball_position, atol=1e-6):
        raise ValueError("Return parameters do not start at the selected service contact.")
    return_result = simulate_racket_impact(return_params, dt=dt, t_max=t_max)
    return ExchangeResult(
        service_params=service_params,
        service_result=service_result,
        contact=contact,
        contact_index=index,
        return_params=return_params,
        return_result=return_result,
        service_validation=validate_service(service_params, service_result, service_targets),
        return_validation=validate_return(return_params, return_result, return_targets),
    )


def _objective_cost(report: ValidationReport, target_tolerance: float, spin_tolerance: float) -> float:
    cost = (report.bounce_error_mm / max(target_tolerance, 1.0)) ** 2
    cost += (report.spin_error_rps / max(spin_tolerance, 1.0)) ** 2
    cost += 10000.0 * len(report.violations)
    return float(cost)


@dataclass(frozen=True)
class _ReturnObjective:
    service_result: SimulationResult
    targets: StrokeTargets
    rubber: RubberProperties
    moment: int
    dt: float
    t_max: float

    def __call__(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=float)
        selection = ContactSelection(moment=self.moment, fraction=float(values[0]))
        _, incoming = contact_state(self.service_result, selection)
        params = RacketImpactParameters(
            ball_velocity=incoming.vel,
            ball_omega=incoming.omega,
            rubber_friction=self.rubber.friction,
            rubber_restitution=self.rubber.restitution,
            racket_angle=(
                -35.0 if self.targets.stroke_side == "forehand" else 35.0,
                float(values[1]),
                float(values[2]),
            ),
            racket_velocity=tuple(float(value) for value in values[3:6]),
            ball_position=incoming.pos,
        )
        result = simulate_racket_impact(params, dt=self.dt, t_max=self.t_max)
        report = validate_return(params, result, self.targets)
        cost = _objective_cost(
            report,
            self.targets.bounce_tolerance_mm,
            self.targets.spin_tolerance_rps,
        )
        post_impact = apply_racket_impact(params)
        cut_stroke = self.targets.spin_rps[1] > 0
        if cut_stroke:
            cost += max(0.0, params.racket_velocity[0] / 1000.0) ** 2 * 100.0
            cost += max(0.0, (params.racket_velocity[2] + 100.0) / 500.0) ** 2 * 100.0
            cost += max(0.0, (post_impact.vel[2] - 1500.0) / 500.0) ** 2 * 100.0
            if params.ball_position[2] < NET_CLEARANCE_LEVEL:
                cost += max(0.0, (100.0 - post_impact.vel[2]) / 500.0) ** 2 * 200.0
        else:
            cost += max(0.0, (100.0 - params.racket_velocity[2]) / 500.0) ** 2 * 100.0
        bounces = table_bounces(result)
        if bounces:
            cost += max(0.0, result.v[0, bounces[0].index] / 500.0) ** 2 * 100.0
        maximum_clearance = (
            self.targets.max_net_clearance_mm
            if self.targets.max_net_clearance_mm is not None
            else (180.0 if cut_stroke else 350.0)
        )
        if np.isfinite(report.net_clearance_mm):
            cost += max(
                0.0,
                (self.targets.min_net_clearance_mm - report.net_clearance_mm) / 50.0,
            ) ** 2 * 200.0
        else:
            cost += 100000.0
        cost += max(0.0, (report.net_clearance_mm - maximum_clearance) / 50.0) ** 2 * 100.0
        return float(cost)


def _service_params_from_vector(
    targets: ServiceTargets,
    rubber: RubberProperties,
    vector: Iterable[float],
) -> RacketImpactParameters:
    values = np.asarray(tuple(vector), dtype=float)
    incoming = InitialConditions(
        pos=(-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
        vel=(0.0, 0.0, -2500.0),
        omega=(0.0, 0.0, 0.0),
    )
    params = racket_parameters_for_target_spin(
        incoming,
        values[2:5],
        values[0],
        values[1],
        rubber,
        "backhand",
    )
    return RacketImpactParameters(
        ball_velocity=params.ball_velocity,
        ball_omega=params.ball_omega,
        rubber_friction=params.rubber_friction,
        rubber_restitution=params.rubber_restitution,
        racket_angle=(80.0, params.racket_angle[1], params.racket_angle[2]),
        racket_velocity=params.racket_velocity,
        ball_position=params.ball_position,
    )


@dataclass(frozen=True)
class _ServiceObjective:
    targets: ServiceTargets
    rubber: RubberProperties
    dt: float
    t_max: float

    def __call__(self, vector: np.ndarray) -> float:
        params = _service_params_from_vector(self.targets, self.rubber, vector)
        result = simulate_racket_impact(params, dt=self.dt, t_max=self.t_max)
        report = validate_service(params, result, self.targets)
        return _objective_cost(
            report,
            self.targets.bounce_tolerance_mm,
            self.targets.spin_tolerance_rps,
        )


def _fallback_differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    maxiter: int,
    popsize: int,
    seed: int,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    dimension = len(bounds)
    population_size = max(12, popsize * dimension)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    if initial_guess is not None:
        population[0] = np.clip(np.asarray(initial_guess, dtype=float), lower, upper)
    scores = np.asarray([objective(candidate) for candidate in population])
    for _ in range(maxiter):
        for index in range(population_size):
            candidates = np.delete(np.arange(population_size), index)
            a_index, b_index, c_index = rng.choice(candidates, 3, replace=False)
            mutant = np.clip(
                population[a_index] + 0.8 * (population[b_index] - population[c_index]),
                lower,
                upper,
            )
            mask = rng.random(dimension) < 0.85
            mask[rng.integers(dimension)] = True
            trial = np.where(mask, mutant, population[index])
            score = objective(trial)
            if score < scores[index]:
                population[index] = trial
                scores[index] = score
    best = int(np.argmin(scores))
    return population[best], float(scores[best])


def _run_search(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    config: ReturnSearchConfig,
    seed: int,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str, str]:
    if differential_evolution is None:
        vector, cost = _fallback_differential_evolution(
            objective,
            bounds,
            config.maxiter,
            config.popsize,
            seed,
            initial_guess=initial_guess,
        )
        message = "SciPy is unavailable; used the built-in differential-evolution fallback."
        if config.workers != 1:
            message += " The fallback is serial, so workers was accepted but not parallelized."
        return vector, cost, "fallback_differential_evolution", message

    result = differential_evolution(
        objective,
        bounds,
        maxiter=config.maxiter,
        popsize=config.popsize,
        seed=seed,
        workers=config.workers,
        updating="deferred" if config.workers != 1 else "immediate",
        polish=False,
        x0=None if initial_guess is None else np.asarray(initial_guess, dtype=float),
    )
    vector = np.asarray(result.x, dtype=float)
    cost = float(result.fun)
    message = str(result.message)
    if config.polish and minimize is not None:
        polished = minimize(objective, vector, method="Powell", bounds=bounds)
        if float(polished.fun) < cost:
            vector = np.asarray(polished.x, dtype=float)
            cost = float(polished.fun)
            message += " Polished with bounded Powell."
    return vector, cost, "scipy.differential_evolution", message


def search_return(
    service_result: SimulationResult,
    targets: StrokeTargets,
    contact: ContactSelection = ContactSelection(),
    rubber: RubberProperties = RubberProperties(),
    config: ReturnSearchConfig = ReturnSearchConfig(),
    use_validated_preset: bool = True,
) -> ReturnSearchResult:
    spin = np.asarray(targets.spin_rps, dtype=float)
    if contact.moment == 3:
        fraction_bounds = (0.499999, 0.500001)
    elif contact.moment == 4 and spin[1] > 0:
        fraction_bounds = (0.50, 0.85)
    else:
        fraction_bounds = (0.0, 1.0)
    bounds = [
        fraction_bounds,
        (-85.0, 85.0),
        (-180.0, 180.0),
        (-14000.0, -200.0),
        (-12000.0, 12000.0),
        (-10000.0, -100.0) if spin[1] > 0 else (100.0, 10000.0),
    ]
    objective = _ReturnObjective(
        service_result=service_result,
        targets=targets,
        rubber=rubber,
        moment=contact.moment,
        dt=config.dt,
        t_max=config.t_max,
    )
    preset_name = None
    if use_validated_preset and contact.moment == 4 and targets.direction == "elbow":
        if spin[1] > 0:
            preset_name = {
                "short": "cut_short",
                "two_bounce": "cut_two_bounce",
                "long": "cut_long",
            }[targets.depth]
            preset_spin = np.array([-15.0, 35.0, 10.0])
        else:
            preset_name = {
                "two_bounce": "top_two_bounce",
                "long": "top_long",
            }.get(targets.depth)
            preset_spin = np.array([0.0, -45.0, 0.0])
        if preset_name is not None and np.linalg.norm(spin - preset_spin) > targets.spin_tolerance_rps:
            preset_name = None
    initial_guess = (
        np.asarray(RETURN_PRESET_VECTORS[preset_name], dtype=float)
        if preset_name is not None
        else None
    )
    if initial_guess is not None:
        selection = ContactSelection(moment=4, fraction=float(initial_guess[0]))
        index, incoming = contact_state(service_result, selection)
        params = RacketImpactParameters(
            ball_velocity=incoming.vel,
            ball_omega=incoming.omega,
            rubber_friction=rubber.friction,
            rubber_restitution=rubber.restitution,
            racket_angle=(
                -35.0 if targets.stroke_side == "forehand" else 35.0,
                float(initial_guess[1]),
                float(initial_guess[2]),
            ),
            racket_velocity=tuple(float(value) for value in initial_guess[3:6]),
            ball_position=incoming.pos,
        )
        trajectory = simulate_racket_impact(params, dt=config.dt, t_max=config.t_max)
        validation = validate_return(params, trajectory, targets)
        if validation.passed:
            return ReturnSearchResult(
                success=True,
                cost=float(objective(initial_guess)),
                vector=tuple(float(value) for value in initial_guess),
                params=params,
                trajectory=trajectory,
                validation=validation,
                contact=selection,
                contact_index=index,
                optimizer="validated_preset",
                message=f"Loaded validated warm start: {preset_name}.",
            )
    best = None
    messages = []
    for restart in range(config.restarts):
        vector, cost, optimizer, message = _run_search(
            objective,
            bounds,
            config,
            config.seed + restart * 1009,
            initial_guess=initial_guess,
        )
        selection = ContactSelection(moment=contact.moment, fraction=float(vector[0]))
        index, incoming = contact_state(service_result, selection)
        params = RacketImpactParameters(
            ball_velocity=incoming.vel,
            ball_omega=incoming.omega,
            rubber_friction=rubber.friction,
            rubber_restitution=rubber.restitution,
            racket_angle=(
                -35.0 if targets.stroke_side == "forehand" else 35.0,
                float(vector[1]),
                float(vector[2]),
            ),
            racket_velocity=tuple(float(value) for value in vector[3:6]),
            ball_position=incoming.pos,
        )
        trajectory = simulate_racket_impact(params, dt=config.dt, t_max=config.t_max)
        validation = validate_return(params, trajectory, targets)
        candidate = ReturnSearchResult(
            success=validation.passed,
            cost=cost,
            vector=tuple(float(value) for value in vector),
            params=params,
            trajectory=trajectory,
            validation=validation,
            contact=selection,
            contact_index=index,
            optimizer=optimizer,
            message=message,
        )
        messages.append(message)
        if best is None or candidate.cost < best.cost:
            best = candidate
        if candidate.success:
            return candidate
    assert best is not None
    best.message = " ".join(messages) + " No restart met every acceptance rule."
    return best


def search_service(
    targets: ServiceTargets = ServiceTargets(),
    rubber: RubberProperties = RubberProperties(friction=1.2, restitution=0.8),
    config: ReturnSearchConfig = ReturnSearchConfig(),
) -> ReturnSearchResult:
    """Search the reverse-pendulum service while keeping toss and rubber fixed."""

    spin = np.asarray(targets.spin_rps, dtype=float)
    tolerance = targets.spin_tolerance_rps
    bounds = [
        (0.0, 2 * np.pi),
        (-24000.0, 24000.0),
        (max(-100.0, spin[0] - tolerance), min(100.0, spin[0] + tolerance)),
        (max(-100.0, spin[1] - tolerance), min(100.0, spin[1] + tolerance)),
        (max(-100.0, spin[2] - tolerance), min(100.0, spin[2] + tolerance)),
    ]
    objective = _ServiceObjective(
        targets=targets,
        rubber=rubber,
        dt=config.dt,
        t_max=config.t_max,
    )
    best = None
    messages = []
    for restart in range(config.restarts):
        vector, cost, optimizer, message = _run_search(
            objective,
            bounds,
            config,
            config.seed + restart * 1009,
        )
        params = _service_params_from_vector(targets, rubber, vector)
        trajectory = simulate_racket_impact(params, dt=config.dt, t_max=config.t_max)
        validation = validate_service(params, trajectory, targets)
        candidate = ReturnSearchResult(
            success=validation.passed,
            cost=cost,
            vector=tuple(float(value) for value in vector),
            params=params,
            trajectory=trajectory,
            validation=validation,
            optimizer=optimizer,
            message=message,
        )
        messages.append(message)
        if best is None or candidate.cost < best.cost:
            best = candidate
        if candidate.success:
            return candidate
    assert best is not None
    best.message = " ".join(messages) + " No restart met every acceptance rule."
    return best
