"""Search table-tennis serve parameters under trajectory constraints.

This module is intentionally heavier than the benchmark scripts. It is meant for
interactive tuning and can use SciPy optimizers when they are available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import benchmark_direct_services as benchmark_direct
import benchmark_racket_services as benchmark_racket
from table_tennis_simulation import (
    BALL_RADIUS,
    InitialConditions,
    RacketImpactParameters,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
    apply_racket_impact,
    simulate,
    simulate_racket_impact,
)


try:
    from scipy.optimize import differential_evolution, minimize
except Exception:  # pragma: no cover - exercised only when SciPy is absent.
    differential_evolution = None
    minimize = None


RPS = 2 * np.pi
TABLE_LEVEL = TABLE_HEIGHT + BALL_RADIUS


@dataclass(frozen=True)
class ServeTargets:
    """Targets and hard-ish constraints for the optimized serve."""

    server_bounce_x: float = 520.0
    server_bounce_y: float = TABLE_WIDTH / 2
    opponent_bounce_x: float = TABLE_LENGTH / 2 + 520.0
    opponent_bounce_y: float = TABLE_WIDTH / 2
    max_height_after_net: float = 260.0
    min_height_after_net: float = 120.0
    require_server_bounce: bool = True
    require_opponent_bounce: bool = True
    prefer_second_opponent_bounce: bool = False
    forbid_second_opponent_bounce: bool = False


@dataclass(frozen=True)
class SearchWeights:
    """Objective-function weights."""

    server_x: float = 1.0
    server_y: float = 0.25
    opponent_x: float = 1.0
    opponent_y: float = 1.0
    max_height: float = 6.0
    min_height: float = 0.6
    bounce_requirement: float = 5000.0
    second_bounce: float = 30.0
    speed_regularization: float = 0.02
    spin_regularization: float = 0.01


@dataclass(frozen=True)
class SearchSpace:
    """Bounds for direct initial-condition optimization."""

    pos_x: tuple[float, float] = (-500.0, -50.0)
    pos_y: tuple[float, float] = (0.0, TABLE_WIDTH)
    pos_z: tuple[float, float] = (TABLE_HEIGHT + 140.0, TABLE_HEIGHT + 380.0)
    vel_x: tuple[float, float] = (3500.0, 12500.0)
    vel_y: tuple[float, float] = (-5200.0, 5200.0)
    vel_z: tuple[float, float] = (-4800.0, -500.0)
    spin_x_rps: tuple[float, float] = (-80.0, 80.0)
    spin_y_rps: tuple[float, float] = (-80.0, 10.0)
    spin_z_rps: tuple[float, float] = (-80.0, 80.0)

    def names(self) -> list[str]:
        return list(self.__dataclass_fields__)

    def bounds(self) -> list[tuple[float, float]]:
        return [getattr(self, name) for name in self.names()]


@dataclass(frozen=True)
class RacketSearchSpace:
    """Bounds for racket-impact parameter optimization."""

    ball_x: tuple[float, float] = (-500.0, -50.0)
    ball_y: tuple[float, float] = (0.0, TABLE_WIDTH)
    ball_z: tuple[float, float] = (TABLE_HEIGHT + 140.0, TABLE_HEIGHT + 380.0)
    incoming_vz: tuple[float, float] = (-5600.0, -900.0)
    pre_spin_x_rps: tuple[float, float] = (-100.0, 100.0)
    pre_spin_y_rps: tuple[float, float] = (-100.0, 100.0)
    pre_spin_z_rps: tuple[float, float] = (-100.0, 100.0)
    friction: tuple[float, float] = (0.35, 1.05)
    restitution: tuple[float, float] = (0.55, 1.05)
    angle_x: tuple[float, float] = (-80.0, 85.0)
    angle_y: tuple[float, float] = (-70.0, 5.0)
    angle_z: tuple[float, float] = (-85.0, 85.0)
    racket_vx: tuple[float, float] = (2000.0, 14000.0)
    racket_vy: tuple[float, float] = (-9000.0, 9000.0)
    racket_vz: tuple[float, float] = (-9000.0, 4000.0)

    def names(self) -> list[str]:
        return list(self.__dataclass_fields__)

    def bounds(self) -> list[tuple[float, float]]:
        return [getattr(self, name) for name in self.names()]


@dataclass
class SearchConfig:
    targets: ServeTargets = field(default_factory=ServeTargets)
    weights: SearchWeights = field(default_factory=SearchWeights)
    space: SearchSpace = field(default_factory=SearchSpace)
    dt: float = 0.005
    t_max: float = 3.0
    maxiter: int = 80
    popsize: int = 10
    polish: bool = True
    seed: int | None = 7
    workers: int = 1


@dataclass
class RacketSearchConfig:
    targets: ServeTargets = field(default_factory=ServeTargets)
    weights: SearchWeights = field(default_factory=SearchWeights)
    space: RacketSearchSpace = field(default_factory=RacketSearchSpace)
    dt: float = 0.005
    t_max: float = 3.0
    maxiter: int = 80
    popsize: int = 10
    polish: bool = True
    seed: int | None = 7
    workers: int = 1


@dataclass
class TrajectoryMetrics:
    server_bounce: tuple[float, float] | None
    opponent_bounce: tuple[float, float] | None
    second_opponent_bounce: tuple[float, float] | None
    max_height_after_net: float
    total_bounces: int
    opponent_bounces: int


@dataclass
class SearchResult:
    success: bool
    cost: float
    optimizer: str
    parameters: dict[str, float | tuple[float, float, float]]
    metrics: TrajectoryMetrics
    message: str = ""

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = asdict(self.metrics)
        return data


@dataclass(frozen=True)
class SearchProgress:
    """Progress update emitted by global search and local polishing."""

    phase: str
    current: int
    total: int
    best_cost: float | None = None
    message: str = ""


def _table_bounce_indices(result) -> np.ndarray:
    contacts = np.isclose(result.x[2], TABLE_LEVEL, atol=1e-6) & (result.v[2] > 0)
    starts = contacts & np.concatenate(([True], ~contacts[:-1]))
    return np.where(starts)[0]


def trajectory_metrics(result) -> TrajectoryMetrics:
    bounces = _table_bounce_indices(result)
    server = [i for i in bounces if result.x[0, i] < TABLE_LENGTH / 2]
    opponent = [i for i in bounces if result.x[0, i] >= TABLE_LENGTH / 2]
    after_net = np.where(result.x[0] >= TABLE_LENGTH / 2)[0]
    max_height = (
        float(np.max(result.x[2, after_net] - TABLE_LEVEL))
        if len(after_net)
        else float("inf")
    )

    def point(index: int) -> tuple[float, float]:
        return float(result.x[0, index]), float(result.x[1, index])

    return TrajectoryMetrics(
        server_bounce=point(server[0]) if server else None,
        opponent_bounce=point(opponent[0]) if opponent else None,
        second_opponent_bounce=point(opponent[1]) if len(opponent) > 1 else None,
        max_height_after_net=max_height,
        total_bounces=int(len(bounces)),
        opponent_bounces=int(len(opponent)),
    )


def _squared_scaled_error(value: float, target: float, scale: float) -> float:
    return ((value - target) / scale) ** 2


def objective_from_metrics(
    metrics: TrajectoryMetrics,
    targets: ServeTargets,
    weights: SearchWeights,
    parameter_vector: Iterable[float],
) -> float:
    cost = 0.0
    vector = np.asarray(list(parameter_vector), dtype=float)

    if metrics.server_bounce is None:
        if targets.require_server_bounce:
            cost += weights.bounce_requirement
    else:
        sx, sy = metrics.server_bounce
        cost += weights.server_x * _squared_scaled_error(sx, targets.server_bounce_x, 100.0)
        cost += weights.server_y * _squared_scaled_error(sy, targets.server_bounce_y, 160.0)

    if metrics.opponent_bounce is None:
        if targets.require_opponent_bounce:
            cost += weights.bounce_requirement
    else:
        ox, oy = metrics.opponent_bounce
        cost += weights.opponent_x * _squared_scaled_error(ox, targets.opponent_bounce_x, 120.0)
        cost += weights.opponent_y * _squared_scaled_error(oy, targets.opponent_bounce_y, 110.0)

    if metrics.max_height_after_net > targets.max_height_after_net:
        cost += weights.max_height * _squared_scaled_error(
            metrics.max_height_after_net,
            targets.max_height_after_net,
            35.0,
        )
    if metrics.max_height_after_net < targets.min_height_after_net:
        cost += weights.min_height * _squared_scaled_error(
            metrics.max_height_after_net,
            targets.min_height_after_net,
            35.0,
        )

    has_second = metrics.second_opponent_bounce is not None
    if targets.prefer_second_opponent_bounce and not has_second:
        cost += weights.second_bounce
    if targets.forbid_second_opponent_bounce and has_second:
        cost += weights.second_bounce

    if len(vector) >= 6:
        cost += weights.speed_regularization * float(np.sum((vector[3:6] / 10000.0) ** 2))
    if len(vector) >= 9:
        cost += weights.spin_regularization * float(np.sum((vector[6:9] / 100.0) ** 2))
    return float(cost)


def direct_vector_to_initial_conditions(vector: Iterable[float]) -> InitialConditions:
    (
        pos_x,
        pos_y,
        pos_z,
        vel_x,
        vel_y,
        vel_z,
        spin_x_rps,
        spin_y_rps,
        spin_z_rps,
    ) = [float(value) for value in vector]
    return InitialConditions(
        pos=(pos_x, pos_y, pos_z),
        vel=(vel_x, vel_y, vel_z),
        omega=(spin_x_rps * RPS, spin_y_rps * RPS, spin_z_rps * RPS),
    )


def initial_conditions_to_direct_vector(ic: InitialConditions) -> np.ndarray:
    return np.array(
        [
            *ic.pos,
            *ic.vel,
            ic.omega[0] / RPS,
            ic.omega[1] / RPS,
            ic.omega[2] / RPS,
        ],
        dtype=float,
    )


def racket_vector_to_parameters(vector: Iterable[float]) -> RacketImpactParameters:
    (
        ball_x,
        ball_y,
        ball_z,
        incoming_vz,
        spin_x_rps,
        spin_y_rps,
        spin_z_rps,
        friction,
        restitution,
        angle_x,
        angle_y,
        angle_z,
        racket_vx,
        racket_vy,
        racket_vz,
    ) = [float(value) for value in vector]
    return RacketImpactParameters(
        ball_position=(ball_x, ball_y, ball_z),
        ball_velocity=(0.0, 0.0, incoming_vz),
        ball_omega=(spin_x_rps * RPS, spin_y_rps * RPS, spin_z_rps * RPS),
        rubber_friction=friction,
        rubber_restitution=restitution,
        racket_angle=(angle_x, angle_y, angle_z),
        racket_velocity=(racket_vx, racket_vy, racket_vz),
    )


def parameters_to_racket_vector(params: RacketImpactParameters) -> np.ndarray:
    return np.array(
        [
            *params.ball_position,
            params.ball_velocity[2],
            params.ball_omega[0] / RPS,
            params.ball_omega[1] / RPS,
            params.ball_omega[2] / RPS,
            params.rubber_friction,
            params.rubber_restitution,
            *params.racket_angle,
            *params.racket_velocity,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class _DirectObjective:
    config: SearchConfig

    def __call__(self, vector: np.ndarray) -> float:
        ic = direct_vector_to_initial_conditions(vector)
        result = simulate(ic, dt=self.config.dt, t_max=self.config.t_max)
        return objective_from_metrics(
            trajectory_metrics(result),
            self.config.targets,
            self.config.weights,
            vector,
        )


@dataclass(frozen=True)
class _RacketObjective:
    config: RacketSearchConfig

    def __call__(self, vector: np.ndarray) -> float:
        params = racket_vector_to_parameters(vector)
        result = simulate_racket_impact(params, dt=self.config.dt, t_max=self.config.t_max)
        return objective_from_metrics(
            trajectory_metrics(result),
            self.config.targets,
            self.config.weights,
            vector,
        )


def _fallback_random_search(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    seed: int | None,
    iterations: int,
    progress_callback: Callable[[SearchProgress], None] | None = None,
) -> tuple[np.ndarray, float, str]:
    rng = np.random.default_rng(seed)
    best_vector = np.array([np.mean(bound) for bound in bounds], dtype=float)
    best_cost = objective(best_vector)
    scale = np.array([bound[1] - bound[0] for bound in bounds], dtype=float)
    center = best_vector.copy()
    rounds = max(1, iterations // 20)
    sample_counts = [80] + [20] * max(0, rounds - 1)
    total_samples = sum(sample_counts)
    completed = 0
    for round_index, samples in enumerate(sample_counts):
        for _ in range(samples):
            if round_index == 0:
                candidate = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
            else:
                candidate = center + rng.normal(0.0, scale * (0.25 ** round_index))
                candidate = np.clip(candidate, [lo for lo, _ in bounds], [hi for _, hi in bounds])
            cost = objective(candidate)
            if cost < best_cost:
                best_vector = candidate
                best_cost = cost
                center = candidate
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    SearchProgress(
                        phase="global",
                        current=completed,
                        total=total_samples,
                        best_cost=float(best_cost),
                        message="Búsqueda global sin SciPy",
                    )
                )
    return best_vector, float(best_cost), "fallback_random_search"


def _run_optimizer(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    maxiter: int,
    popsize: int,
    polish: bool,
    seed: int | None,
    workers: int,
    initial_guess: np.ndarray | None = None,
    progress_callback: Callable[[SearchProgress], None] | None = None,
) -> tuple[np.ndarray, float, str, str]:
    if progress_callback is not None:
        progress_callback(SearchProgress("global", 0, maxiter, message="Iniciando búsqueda global"))
    if differential_evolution is None:
        vector, cost, name = _fallback_random_search(
            objective,
            bounds,
            seed,
            iterations=max(300, maxiter * popsize * len(bounds)),
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(SearchProgress("complete", 1, 1, best_cost=cost, message="Búsqueda terminada"))
        return vector, cost, name, "SciPy not available; used fallback random search."

    generation = 0

    def scipy_progress(xk, convergence):
        nonlocal generation
        generation += 1
        if progress_callback is not None:
            progress_callback(
                SearchProgress(
                    phase="global",
                    current=min(generation, maxiter),
                    total=maxiter,
                    best_cost=float(objective(np.asarray(xk, dtype=float))),
                    message=f"Convergencia: {convergence:.4g}",
                )
            )
        return False

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        seed=seed,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
        callback=scipy_progress,
    )
    vector = np.asarray(result.x, dtype=float)
    cost = float(result.fun)
    message = str(result.message)

    if initial_guess is not None:
        guess_cost = objective(np.asarray(initial_guess, dtype=float))
        if guess_cost < cost:
            vector = np.asarray(initial_guess, dtype=float)
            cost = float(guess_cost)
            message += " Initial guess improved the result."

    if polish and minimize is not None:
        polish_iteration = 0
        polish_total = max(200, maxiter * 20)

        def polish_progress(current_vector):
            nonlocal polish_iteration
            polish_iteration += 1
            if progress_callback is not None:
                progress_callback(
                    SearchProgress(
                        phase="polish",
                        current=min(polish_iteration, polish_total),
                        total=polish_total,
                        best_cost=float(objective(np.asarray(current_vector, dtype=float))),
                        message="Pulido local con Nelder-Mead",
                    )
                )

        polished = minimize(
            objective,
            vector,
            method="Nelder-Mead",
            callback=polish_progress,
            options={"maxiter": polish_total, "xatol": 1e-2, "fatol": 1e-3},
        )
        if float(polished.fun) < cost:
            vector = np.asarray(polished.x, dtype=float)
            vector = np.clip(vector, [lo for lo, _ in bounds], [hi for _, hi in bounds])
            cost = float(objective(vector))
            message += " Polished with Nelder-Mead."

    if progress_callback is not None:
        progress_callback(SearchProgress("complete", 1, 1, best_cost=cost, message="Búsqueda terminada"))
    return vector, cost, "scipy.differential_evolution", message


def search_direct_parameters(
    config: SearchConfig,
    initial_guess: InitialConditions | np.ndarray | None = None,
    progress_callback: Callable[[SearchProgress], None] | None = None,
) -> SearchResult:
    bounds = config.space.bounds()
    guess_vector = None
    if isinstance(initial_guess, InitialConditions):
        guess_vector = initial_conditions_to_direct_vector(initial_guess)
    elif initial_guess is not None:
        guess_vector = np.asarray(initial_guess, dtype=float)

    objective = _DirectObjective(config)

    vector, cost, optimizer, message = _run_optimizer(
        objective,
        bounds,
        config.maxiter,
        config.popsize,
        config.polish,
        config.seed,
        config.workers,
        initial_guess=guess_vector,
        progress_callback=progress_callback,
    )
    ic = direct_vector_to_initial_conditions(vector)
    result = simulate(ic, dt=config.dt, t_max=config.t_max)
    metrics = trajectory_metrics(result)
    return SearchResult(
        success=metrics.server_bounce is not None and metrics.opponent_bounce is not None,
        cost=cost,
        optimizer=optimizer,
        message=message,
        parameters={
            "position": tuple(float(x) for x in ic.pos),
            "velocity": tuple(float(x) for x in ic.vel),
            "spin_rps": tuple(float(x / RPS) for x in ic.omega),
            "omega_rad_s": tuple(float(x) for x in ic.omega),
        },
        metrics=metrics,
    )


def search_racket_parameters(
    config: RacketSearchConfig,
    initial_guess: RacketImpactParameters | np.ndarray | None = None,
    progress_callback: Callable[[SearchProgress], None] | None = None,
) -> SearchResult:
    bounds = config.space.bounds()
    guess_vector = None
    if isinstance(initial_guess, RacketImpactParameters):
        guess_vector = parameters_to_racket_vector(initial_guess)
    elif initial_guess is not None:
        guess_vector = np.asarray(initial_guess, dtype=float)

    objective = _RacketObjective(config)

    vector, cost, optimizer, message = _run_optimizer(
        objective,
        bounds,
        config.maxiter,
        config.popsize,
        config.polish,
        config.seed,
        config.workers,
        initial_guess=guess_vector,
        progress_callback=progress_callback,
    )
    params = racket_vector_to_parameters(vector)
    result = simulate_racket_impact(params, dt=config.dt, t_max=config.t_max)
    post_ic = apply_racket_impact(params)
    metrics = trajectory_metrics(result)
    return SearchResult(
        success=metrics.server_bounce is not None and metrics.opponent_bounce is not None,
        cost=cost,
        optimizer=optimizer,
        message=message,
        parameters={
            "ball_position": tuple(float(x) for x in params.ball_position),
            "incoming_ball_velocity": tuple(float(x) for x in params.ball_velocity),
            "preimpact_spin_rps": tuple(float(x / RPS) for x in params.ball_omega),
            "rubber_friction": float(params.rubber_friction),
            "rubber_restitution": float(params.rubber_restitution),
            "racket_angle": tuple(float(x) for x in params.racket_angle),
            "racket_velocity": tuple(float(x) for x in params.racket_velocity),
            "postimpact_velocity": tuple(float(x) for x in post_ic.vel),
            "postimpact_spin_rps": tuple(float(x / RPS) for x in post_ic.omega),
        },
        metrics=metrics,
    )


def benchmark_initial_guess(service: str, depth: str, lane: str) -> InitialConditions:
    for case in benchmark_direct.build_cases():
        if (case.service, case.depth, case.lane) == (service, depth, lane):
            return case.initial_conditions
    raise ValueError(f"No direct benchmark case for {(service, depth, lane)}")


def benchmark_racket_initial_guess(service: str, depth: str, lane: str) -> RacketImpactParameters:
    for case in benchmark_racket.build_cases():
        if (case.service, case.depth, case.lane) == (service, depth, lane):
            return case.params
    raise ValueError(f"No racket benchmark case for {(service, depth, lane)}")


def target_from_benchmark(depth: str, lane: str, server_x: float | None = None) -> ServeTargets:
    if depth not in benchmark_direct.DEPTHS:
        raise ValueError(f"Unknown depth: {depth}")
    if lane not in benchmark_direct.LANES:
        raise ValueError(f"Unknown lane: {lane}")
    return ServeTargets(
        server_bounce_x=server_x if server_x is not None else {"short": 440.0, "two_bounce": 530.0, "long": 580.0}[depth],
        server_bounce_y=benchmark_direct.LANES[lane]["y"],
        opponent_bounce_x=benchmark_direct.DEPTHS[depth]["target_x"],
        opponent_bounce_y=benchmark_direct.LANES[lane]["y"],
        max_height_after_net={"short": 220.0, "two_bounce": 235.0, "long": 250.0}[depth],
        min_height_after_net=120.0,
        prefer_second_opponent_bounce=depth in {"short", "two_bounce"},
        forbid_second_opponent_bounce=depth == "long",
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search serve parameters under trajectory constraints.")
    parser.add_argument("--mode", choices=["direct", "racket"], default="direct")
    parser.add_argument("--service", default="pendulum", choices=list(benchmark_direct.SERVICE_TYPES))
    parser.add_argument("--depth", default="two_bounce", choices=list(benchmark_direct.DEPTHS))
    parser.add_argument("--lane", default="elbow", choices=list(benchmark_direct.LANES))
    parser.add_argument("--server-x", type=float, help="Target first bounce x on server side.")
    parser.add_argument("--opponent-x", type=float, help="Target first bounce x on receiver side.")
    parser.add_argument("--opponent-y", type=float, help="Target first bounce y on receiver side.")
    parser.add_argument("--max-height", type=float, help="Maximum height over ball-table level after net.")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    targets = target_from_benchmark(args.depth, args.lane, args.server_x)
    if args.opponent_x is not None:
        targets = ServeTargets(**{**asdict(targets), "opponent_bounce_x": args.opponent_x})
    if args.opponent_y is not None:
        targets = ServeTargets(**{**asdict(targets), "opponent_bounce_y": args.opponent_y})
    if args.max_height is not None:
        targets = ServeTargets(**{**asdict(targets), "max_height_after_net": args.max_height})

    if args.mode == "direct":
        config = SearchConfig(
            targets=targets,
            dt=args.dt,
            t_max=args.t_max,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=not args.no_polish,
            seed=args.seed,
            workers=args.workers,
        )
        result = search_direct_parameters(
            config,
            initial_guess=benchmark_initial_guess(args.service, args.depth, args.lane),
        )
    else:
        config = RacketSearchConfig(
            targets=targets,
            dt=args.dt,
            t_max=args.t_max,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=not args.no_polish,
            seed=args.seed,
            workers=args.workers,
        )
        result = search_racket_parameters(
            config,
            initial_guess=benchmark_racket_initial_guess(args.service, args.depth, args.lane),
        )

    payload = result.to_json_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
